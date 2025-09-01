import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate
from tqdm import tqdm
# from evaluation.eval_utils import *
from utils_general import progress_bar


class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

    @torch.no_grad()
    def evaluate(self, rollouts):
        # identical preprocessing
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        self.actor_critic.eval()
        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        denom = 0

        data_generator = rollouts.feed_forward_generator(advantages, self.mini_batch_size)
        for sample in data_generator:
            (obs_batch, output_ids_batch, actions_batch,
             value_preds_batch, return_batch, masks_batch,
             old_action_log_probs_batch, adv_targ) = sample

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                values, action_log_probs = self.actor_critic.evaluate_actions(**obs_batch[0])

                old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                adv_targ = adv_targ.to(action_log_probs.device)
                value_preds_batch = value_preds_batch.to(values.device)
                return_batch = return_batch.to(values.device)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = (-surr2 if torch.any(ratio > 10) else -torch.min(surr1, surr2)).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            denom += 1

        # don't change training mode here; your update() sets it explicitly
        denom = max(1, denom)
        return value_loss_epoch / denom, action_loss_epoch / denom, 0.0

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch, action_loss_epoch, dist_entropy_epoch = 0, 0, 0
        grad_step = 0
        accumulation_steps = self.accelerator.gradient_accumulation_steps  # ▲

        self.actor_critic.train()
        self.optimizer.zero_grad()

        pbar_ppo = progress_bar(self.ppo_epoch, "PPO Training", "yellow",
                                self.accelerator)

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.mini_batch_size)

            pbar = progress_bar(rollouts.num_steps,
                                f"PPO Training Epoch {e}/{self.ppo_epoch}",
                                "green", self.accelerator)

            for sample in data_generator:
                grad_step += 1

                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    (obs_batch, output_ids_batch, actions_batch,
                     value_preds_batch, return_batch, masks_batch,
                     old_action_log_probs_batch, adv_targ) = sample

                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        **obs_batch[0])  # batch size == 1

                    if torch.isnan(action_log_probs).any():
                        continue

                    old_action_log_probs_batch = old_action_log_probs_batch.to(
                        action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)

                    ratio = torch.exp(action_log_probs -
                                      old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss = (-surr2 if torch.any(ratio > 10)
                                   else -torch.min(surr1, surr2)).mean()

                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + (
                                values - value_preds_batch).clamp(-self.clip_param,
                                                                  self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                                value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                     value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    loss = value_loss * self.value_loss_coef + action_loss
                    self.accelerator.print(
                        f"[Step {grad_step}] "
                        f"Loss: {loss.item():.6f} | "
                        f"Value Loss: {value_loss.item():.6f} | "
                        f"Action Loss: {action_loss.item():.6f}"
                    )


                    # loss.backward()  #
                    self.accelerator.backward(loss)

                # if grad_step % accumulation_steps == 0:  #
                if (grad_step + 1) % accumulation_steps == 0:
                    if self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                pbar.update()

            pbar.close()
            pbar_ppo.update()

        pbar_ppo.close()

        denom = max(1, grad_step)
        return (value_loss_epoch / denom,
                action_loss_epoch / denom,
                dist_entropy_epoch / denom)
