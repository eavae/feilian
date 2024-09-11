from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# 定义groundtruth和prediction列表
groundtruth = [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]
prediction = [["a", "b"], ["a"], ["a", "b", "c"], ["a", "c"], ["c"], []]

# 将列表转化为集合的形式，以便处理多标签分类问题
# 对每一个 groundtruth 和 prediction 元素都转换成集合，方便计算交集、并集等操作
groundtruth_sets = [set(gt) for gt in groundtruth]
prediction_sets = [set(pred) for pred in prediction]

# 计算每个样本的TP（True Positives）、FP（False Positives）和FN（False Negatives）
# 使用列表解析，每个样本的 TP、FP 和 FN 分别通过交集、差集来计算
TP = [len(gt & pred) for gt, pred in zip(groundtruth_sets, prediction_sets)]
FP = [len(pred - gt) for gt, pred in zip(groundtruth_sets, prediction_sets)]
FN = [len(gt - pred) for gt, pred in zip(groundtruth_sets, prediction_sets)]

# Precision: 计算整体的精确率，精确率为 TP/(TP + FP)
# 避免分母为0的情况，TP和FP之和为0时，返回0
precision = sum(TP) / (sum(TP) + sum(FP)) if (sum(TP) + sum(FP)) > 0 else 0

# Recall: 计算整体的召回率，召回率为 TP/(TP + FN)
# 同样避免分母为0的情况
recall = sum(TP) / (sum(TP) + sum(FN)) if (sum(TP) + sum(FN)) > 0 else 0

# Accuracy: 计算整体的准确率，准确率为所有正确预测（TP）的比例
accuracy = sum(TP) / sum([len(gt) for gt in groundtruth_sets])

# F1 Score: F1是precision和recall的调和平均值，避免分母为0的情况
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# 输出结果
print("手动计算的结果：")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print()

## Calculate the precision, recall, accuracy and F1 score by Scikit-learn ###########################################################################
# 手动计算关注的是样本级别的统计量，而 Scikit-learn 的微平均则是基于全局标签的统计
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# 定义groundtruth和prediction列表
groundtruth = [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]
prediction = [["a", "b"], ["a"], ["a", "b", "c"], ["a", "c"], ["c"], []]

# 将groundtruth和prediction进行二值化编码
mlb = MultiLabelBinarizer()  # MultiLabelBinarizer会将标签进行one-hot编码处理
groundtruth_encoded = mlb.fit_transform(groundtruth)
prediction_encoded = mlb.transform(prediction)

# 计算 precision, recall, accuracy 和 F1 score
# 'micro' 表示计算全局的TP, FP, FN，而不是每个标签分别计算后取平均
precision = precision_score(groundtruth_encoded, prediction_encoded, average='micro')
recall = recall_score(groundtruth_encoded, prediction_encoded, average='micro')
f1 = f1_score(groundtruth_encoded, prediction_encoded, average='micro')
accuracy = accuracy_score(groundtruth_encoded, prediction_encoded)

# 输出结果
print("Scikit-learn 计算的结果，micro：")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print()

## Scikit-learn 可以进行样本级别的统计计算。你可以通过设置 average=None 来让 Scikit-learn 返回每个样本或标签的精确率、召回率和 F1 score，而不是汇总的平均值
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# 定义groundtruth和prediction列表
groundtruth = [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]
prediction = [["a", "b"], ["a"], ["a", "b", "c"], ["a", "c"], ["c"], []]

# 将groundtruth和prediction进行二值化编码
mlb = MultiLabelBinarizer()
groundtruth_encoded = mlb.fit_transform(groundtruth)
prediction_encoded = mlb.transform(prediction)

# 计算样本级别的 precision, recall 和 F1 score
precision_per_label = precision_score(groundtruth_encoded, prediction_encoded, average=None)
recall_per_label = recall_score(groundtruth_encoded, prediction_encoded, average=None)
f1_per_label = f1_score(groundtruth_encoded, prediction_encoded, average=None)

# 输出每个标签的 precision, recall 和 F1 score
print("Scikit-learn 计算的结果，per label：")
print(f"Precision per label: {precision_per_label}")
print(f"Recall per label: {recall_per_label}")
print(f"F1 Score per label: {f1_per_label}")

# 计算样本级别的 accuracy
accuracy = accuracy_score(groundtruth_encoded, prediction_encoded)
# 输出总体的 accuracy
print(f"Accuracy: {accuracy:.2f}")


# 手动计算的结果：
# Precision: 0.67
# Recall: 0.50
# Accuracy: 0.50
# F1 Score: 0.57


# Scikit-learn 计算的结果，micro：
# Precision: 1.00
# Recall: 0.50
# F1 Score: 0.67
# Accuracy: 0.33

# Scikit-learn 计算的结果，per label：
# Precision per label: [1. 1.]
# Recall per label: [0.66666667 0.33333333]
# F1 Score per label: [0.8 0.5]
# Accuracy: 0.33

