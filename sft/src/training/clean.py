import json


def remove_outer_brackets(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Original data type:", type(data))
    if isinstance(data, list):
        print("Original list length:", len(data))
    else:
        print("Original data is not a list.")


    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        new_data = data[0]
        print("Detected single-element nested list. New data length:", len(new_data))
    else:
        new_data = data
        final_data = []
        for data_item in data:
            for d in data_item:
                final_data.append(d)
        print("Data structure unchanged.")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(final_data, f_out, indent=4, ensure_ascii=False)

    print(f"Processed file saved to {output_file}")

if __name__ == '__main__':
    input_file = '/mnt/workspace/jinhangzhan/data/SFTvsRL_Data/SFT_Data/gp-l/ood-test-data.json'
    output_file = '/mnt/workspace/jinhangzhan/data/SFTvsRL_Data/SFT_Data/gp-l/ood-test-data-clean.json'
    remove_outer_brackets(input_file, output_file)