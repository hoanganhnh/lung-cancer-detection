import joblib
import numpy as np

model = joblib.load("heightPredict/heightPredict.pkl")


def normalize(value):
    min_value = 0
    max_value = 100
    value = (value - min_value) / (max_value - min_value)
    return value


def predict(father=50, mother=50, gender=0, stt=0):
    pre = np.zeros((34))
    pre[0] = normalize(father)
    pre[1] = normalize(mother)
    pre[2] = gender
    pre[stt + 2 - 1] = 1
    pre = [pre, pre]
    return model.predict(pre)[0] * 100


# chiều cao bố mẹ trong khoảng từ 0 - 100 tương ứng với giá trị chiều cao thực (cm) - 100
fatherHeight = 70.5
motherHeight = 65.5

# 0: Male | 1: Female
gender = 0

# là con thứ mấy > 0 và < 31
stt = 4

print(predict(fatherHeight, motherHeight, 1, 4))
