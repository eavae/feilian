from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# 定义groundtruth和prediction列表
groundtruth = [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]
prediction = [["a", "b"], ["a"], ["a", "b", "c"], ["a", "c"], ["c"], []]

# 将groundtruth和prediction进行二值化编码
mlb = MultiLabelBinarizer()
groundtruth_encoded = mlb.fit_transform(groundtruth)
prediction_encoded = mlb.transform(prediction)

# Micro平均
precision_micro = precision_score(groundtruth_encoded, prediction_encoded, average='micro')
recall_micro = recall_score(groundtruth_encoded, prediction_encoded, average='micro')
accuracy_micro = accuracy_score(groundtruth_encoded, prediction_encoded)

# Macro平均
precision_macro = precision_score(groundtruth_encoded, prediction_encoded, average='macro')
recall_macro = recall_score(groundtruth_encoded, prediction_encoded, average='macro')

# 样本平均
precision_samples = precision_score(groundtruth_encoded, prediction_encoded, average='samples')
recall_samples = recall_score(groundtruth_encoded, prediction_encoded, average='samples')

# 输出结果
print(f"Micro Precision: {precision_micro:.2f}, Micro Recall: {recall_micro:.2f}, Accuracy: {accuracy_micro:.2f}")
print(f"Macro Precision: {precision_macro:.2f}, Macro Recall: {recall_macro:.2f}")
print(f"Samples Precision: {precision_samples:.2f}, Samples Recall: {recall_samples:.2f}")
