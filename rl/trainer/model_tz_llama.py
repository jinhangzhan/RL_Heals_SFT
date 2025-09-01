import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from rl.trainer.interface import *
# from evaluation.eval_utils import *
# from utils_mllm import *
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class VLMValue(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        # h = base.config.hidden_size
        if hasattr(base.config, "hidden_size"):
            h = base.config.hidden_size
        else:
            h = 4096
        # h = get_hidden(base.config)
        self.value_head = nn.Sequential(
            nn.Linear(h, h // 4),
            nn.ReLU(),
            nn.Linear(h // 4, h // 8),
            nn.ReLU(),
            nn.Linear(h // 8, 1)
        ).to(base.device, dtype=torch.bfloat16)

    def forward(self, inputs):
        out = self.base(**inputs, output_hidden_states=True)
        last_token = out.hidden_states[-1][:, -1]  # [B, hidden]
        return self.value_head(last_token)

class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                value_model,
                generation_config,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        # self.args = args
        self.tokenizer = tokenizer
        self.value_model = value_model
        self.base = value_model.base
        # self.INPUT_IDS = INPUT_IDS # dumb solution
        self.temperature = generation_config.temperature
        self.max_new_tokens = generation_config.max_new_tokens
        self.thought_prob_coef = generation_config.thought_prob_coef
        self.token_cnt = 0
        self.called_inference_time = 0
        self.called_bp_time = 0


    def act_oneline(self, inputs, obs=None):
        # count the number of tokens and add to the token_cnt
        self.token_cnt += inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = self.base.generate(
            **inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, 
            output_scores=False,
            output_hidden_states=False,
            return_dict_in_generate=True,
            # pad_token_id=self.tokenizer.eos_token_id
            )
            output_ids = outputs['sequences'][:, inputs['input_ids'].shape[1]:]
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print("output_text:", output_text)
        # output_text_w_special_tokens = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        # print("output_text_w_special_tokens:", output_text_w_special_tokens)
        cated_io = torch.cat((inputs['input_ids'], output_ids), dim = 1)

        # estimate the total input tokens for this inference, given it is an autoregressive process, it is the sum of input_tokens, input_tokens + 1, ..., input_tokens + output_tokens
        # for i in range(inputs['input_ids'].shape[1], cated_io.shape[1]):
        #     self.token_cnt += 2*i
        self.called_inference_time += 1
        # print the shape and assert False
        # print("cated_io:", cated_io.shape)
        # print("inputs['input_ids']:", inputs['input_ids'].shape)
        # assert False
        assert cated_io.shape == outputs['sequences'].shape
        cated_io_txt = self.tokenizer.decode(cated_io[0], skip_special_tokens=False)
        # print("cated_io_txt:", cated_io_txt)
        # assert False
        # llama processor will automatically add one more bos token, which sucks
        new_inputs = {"input_ids": cated_io}
        for key in inputs.keys():
            if key != 'input_ids':
                # if key in ['cross_attention_mask', 'attention_mask']
                if key in ['cross_attention_mask', 'attention_mask']:
                    new_shape = list(inputs[key].shape)
                    new_shape[1] = int(output_ids.shape[1])
                    new_inputs[key] = torch.cat([inputs[key], torch.ones(new_shape, dtype=inputs[key].dtype, device=inputs[key].device)], dim=1)
                else:
                    new_inputs[key] = inputs[key]
        input_ids_register = inputs['input_ids']
        # for key, item in new_inputs.items():
        #     print(f"key: {key}, shape: {item.shape}")
        #     print(f"old key: {key}, shape: {inputs[key].shape}")
        # del inputs
        # print("new_inputs:", new_inputs['input_ids'].shape)
        # print("cated_io:", cated_io.shape)
        # # show the difference between new_inputs and cated_io
        # print("new_inputs:", new_inputs['input_ids'])
        # print("cated_io:", cated_io)

        # # enter a for loop and print out the decoded text, token id
        # for i in range(max(new_inputs['input_ids'].shape[1], cated_io.shape[1])):
        #     try:
        #         print(f"idx: {i}, token_id: {cated_io[0][i]}, decoded_text: {self.tokenizer.decode(cated_io[0][i], skip_special_tokens=False)}")
        #     except:
        #         print(f"ids: {i} out of range of cated_io")
        #     # do the same for new_inputs
        #     try:
        #         print(f"idx: {i}, token_id: {new_inputs['input_ids'][0][i]}, decoded_text: {self.tokenizer.decode(new_inputs['input_ids'][0][i], skip_special_tokens=False)}")
        #     except:
        #         print(f"ids: {i} out of range of new_inputs")
        # # then print remaining tokens
        # print(new_inputs.keys())
        # # print all items in new_inputs
        # for key in new_inputs.keys():
        #     print(f"new key: {key}, value: {new_inputs[key]}, shape: {new_inputs[key].shape}")
        #     # old inputs 
        #     print(f"old key: {key}, value: {inputs[key]}, shape: {inputs[key].shape}")
        #     print("=========")
        assert new_inputs['input_ids'].shape[1] == cated_io.shape[1]
        io_dict = {"io_pair": (input_ids_register, output_ids), **new_inputs}
        with torch.no_grad():
            values, sum_log_prob, action_tokens_log_prob = self.evaluate(**io_dict, inference=True)
        return values, io_dict, output_text, sum_log_prob, action_tokens_log_prob
    
    def cat_io_pair(self, io_pair):
        to_be_cated = []
        # for ele in io_lst:
        # ele = io_lst
        input_ids, output_ids = io_pair[0], io_pair[1]
        to_be_cated.append(input_ids.to(self.base.device))
        to_be_cated.append(output_ids.to(self.base.device))
        return torch.cat(to_be_cated, dim = 1)
    
    def evaluate(self, io_pair, inference=False,**kwargs):

        cated_io = kwargs['input_ids']
        input_ids = io_pair[0]
        output_ids = cated_io[:, input_ids.shape[1]:]


        outputs= self.base(
            output_hidden_states = True,
            **kwargs
            )
        if inference:
            # self.token_cnt += 2 * cated_io.shape[1]
            pass
        else:
            if 'pixel_values' in kwargs.keys():
                self.token_cnt += 6 * (cated_io.shape[1]+1600) # llama 3.2 has 1601 visual tokens
                self.called_bp_time += 1
            else:
                self.token_cnt += 6 * cated_io.shape[1]
                self.called_bp_time += 1
        
        scores = outputs.logits

        input_token_len = input_ids.shape[1]
        try:
            hidden_states = outputs.hidden_states[-1][:, input_token_len-1].to(torch.bfloat16)
            values = self.value_model.value_head(hidden_states)
            scores = scores * (1/self.temperature)
            scores = scores.to(torch.float32)
            log_probs = torch.nn.functional.log_softmax(scores, dim=-1)

            log_probs = log_probs.to(torch.bfloat16)
            output_ids_mask = (output_ids != 0)[:, 1:]

            selected_log_probs = output_ids_mask * torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
            # assert False, "Debug action"
            unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
        # except:
        except Exception as e:
            print(f"A general exception occurred: {e}")
            # in this case, usually the model refuse to reply, or generate fewer tokens than 3.
            print("outputs.hidden_states:", outputs.hidden_states[-1].shape)
            print("input_token_len:", input_token_len)
            print("input_ids:", input_ids.shape)
            print("output_ids:", output_ids.shape)
            print("cated_io:", cated_io.shape)
            # decode cated io and input ids
            decoded_io = self.tokenizer.decode(cated_io[0], skip_special_tokens=False)
            decoded_input_ids = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print("decoded_io:", decoded_io)
            print("#####################")
            print(cated_io)
            print("#####################")
            print("decoded_input_ids:", decoded_input_ids)
            print("#####################")
            print(input_ids)
            print("#####################")
            dummy = torch.zeros(values.size(0), device=self.base.device, dtype=values.dtype)
            return values, dummy, dummy
            # assert False

        
        map_dict = []
        try:
            for i in range(len(output_ids[0])-1):
                map_dict.append({self.tokenizer.batch_decode(output_ids[:,i:i+1], skip_special_tokens=True)[0]: output_ids[:,i:i+1]})
            for i, dic in enumerate(map_dict):
                if 'formula' in dic.keys():
                    target = torch.cat((list(map_dict[i-1].values())[0], list(map_dict[i].values())[0], list(map_dict[i+1].values())[0]), dim=0).flatten()
                elif 'action' in dic.keys():
                    target = torch.tensor([330, 1335, 794]).to(self.base.device)
                    break
                    # print(target)
            assert target
           
        except:
            target = torch.tensor([330,60599,794]).to(self.base.device)
        # print("target:", target)
        target = target.to(self.base.device)
        # assert False, "Debug action"
        # target = torch.tensor([29908,2467,1115]).to(base.device)

        # tokens for text string:'"action":' (torch.tensor([[29908,2467,1115]]))
        matches = (unfolded == target).all(dim = -1)
        # print("matches:", matches)
        match_index = matches.nonzero(as_tuple=True)[-1]
        # print("match_index:", match_index)
        # print("selected_log_prob:", selected_log_probs.shape, selected_log_probs)
        if match_index.shape[0] >= 1:
            match_index = match_index[-1].unsqueeze(0)
        else:
            try:
                match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
            except:
                sum_log_prob = torch.tensor([-2]).to(self.base.device)
                action_tokens_log_prob = torch.tensor([-1]).to(self.base.device)
                return values, sum_log_prob, action_tokens_log_prob
        
        # print("output_ids_mask:", output_ids_mask)
        # print("match_index_2", match_index)
        ## omitting the second token for calculating log prob, because its logprb is very very small
        
        # decode cated io
        # decoded_io = self.tokenizer.decode(cated_io[0], skip_special_tokens=False)
        thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
        
        # print all items of selected log probs
        # for i in range(selected_log_probs.shape[1]):
        #     # if not zero
        #     if torch.sum(selected_log_probs[:,i]) != 0:
                
        #         print(f"selected_log_probs {i}:", selected_log_probs[:,i])
            # print("selected_log_probs:", selected_log_probs[:,i])
            
        action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
        # print("action_tokens_log_prob:", action_tokens_log_prob)
        sum_log_prob = self.thought_prob_coef*thought_log_prob + action_tokens_log_prob
        return values, sum_log_prob, action_tokens_log_prob


    def evaluate_actions(self, io_dict, image = None):
        value, action_log_prob, _ = self.evaluate(**io_dict)
        return value, action_log_prob