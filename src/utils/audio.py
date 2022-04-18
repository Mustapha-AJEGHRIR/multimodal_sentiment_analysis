# def speech_file_to_array_fn(path):
#     speech_array, sampling_rate = torchaudio.load(path)
#     resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
#     speech = resampler(speech_array).squeeze().numpy()
#     return speech

# def label_to_id(label, label_list):

#     if len(label_list) > 0:
#         return label_list.index(label) if label in label_list else -1

#     return label

# def preprocess_function(examples):
#     speech_list = [speech_file_to_array_fn(path) for path in examples["path"]]
#     target_list = [label_to_id(label, label_list) for label in examples["emotion"]]

#     result = processor(speech_list, sampling_rate=target_sampling_rate)
#     result["labels"] = list(target_list)
#     # print(result.keys())

#     return result
