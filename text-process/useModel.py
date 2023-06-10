import joblib
import numpy as np

model = joblib.load("New Folder/Models/randomForest_model.pkl")


def predict(
    gender=0,
    age_group=0,
    occupice=0,
    distance=0,
    cooking=0,
    air_pollution=0,
    cig_smoke=0,
    smoking=0,
    herbicide=0,
    insecticides=0,
):
    age_group_arr = np.zeros((4))
    age_group_arr[age_group] = 1
    distance_arr = np.zeros((3))
    distance_arr[distance] = 1
    cig_smoke_arr = np.zeros((3))
    cig_smoke_arr[cig_smoke] = 1
    distance_arr = np.zeros((3))
    distance_arr[distance] = 1
    status = [
        gender,
        age_group_arr[0],
        age_group_arr[1],
        age_group_arr[2],
        age_group_arr[3],
        occupice,
        distance_arr[0],
        distance_arr[1],
        distance_arr[2],
        cooking,
        air_pollution,
        cig_smoke_arr[0],
        cig_smoke_arr[1],
        cig_smoke_arr[2],
        distance_arr[0],
        distance_arr[1],
        distance_arr[2],
        herbicide,
        insecticides,
    ]
    status = [status, status]
    return model.predict(status)[0]


# 0 is Female 1 is Male
gender = 0
# age group
# 0: < 58
# 1: 55 - 64
# 2: 65 - 75
# 3: >= 75
age_group = 0


# là nhà nông
occupice = 0

# distance from crop
# 0: < 500
# 1: 500 - 1000
# 2: > 1000
distance = 1


# thường xuyên tiếp xúc với bếp
# 0: không, ít khi
# 1: thường xuyên, nhiều
cooking = 0

# Trong môi trường ô nhiễm
# 0: rừng, làng quê ít bị ô nhiễm
# 1: thành thị, khu công nghiệp, nhà máy
air_pollution = 1

# số lượng thuốc lá đã hút
# 0: chưa bao h hút hoặc hút rất ít
# 1: ít hơn 109500 điếu
# 2: trên 109500 điếu
cig_smoke = 0


# tình trạng hút thuốc hiện tại
# 0: không hoặc chưa bao h hút
# 1: đã từng hút, bây h bỏ
# 2: đang hút thuốc
smoking = 0

# thường xuyên Tiếp xúc với thuốc trừ sâu
herbicide = 0

# thường xuyên Tiếp xúc với thuốc diệt cỏ
insecticides = 0

print(
    predict(
        gender,
        age_group,
        occupice,
        distance,
        cooking,
        air_pollution,
        cig_smoke,
        smoking,
        herbicide,
        insecticides,
    )
)
