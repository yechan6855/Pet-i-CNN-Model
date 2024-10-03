import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# 1. 데이터 불러오기
data = pd.read_csv('./dataset/DogMoveData.csv')

# 2. 제거할 행동 리스트
remove_behaviors = ['Bowing', 'Jumping', 'Tugging',
                    'Synchronization', 'Extra_Synchronization', 'Lying chest',
                    'Sniffing', 'Playing', 'Panting', 'Eating', 'Pacing',
                    'Drinking', 'Carrying object', '<undefined>']

# 3. 제거할 행동을 제외한 데이터 필터링
filtered_movedata = data[~data['Behavior_1'].isin(remove_behaviors)]

# 4. Task별로 각 행동에 맞는 데이터만 남기고 나머지 제거
# 각 행동에 맞는 Task를 남기고, 나머지를 제거
filter_conditions = (
    (filtered_movedata['Behavior_1'] == 'Walking') & (filtered_movedata['Task'] == 'Task walk') |
    (filtered_movedata['Behavior_1'] == 'Trotting') & (filtered_movedata['Task'] == 'Task trot') |
    (filtered_movedata['Behavior_1'] == 'Sitting') & (filtered_movedata['Task'] == 'Task sit') |
    (filtered_movedata['Behavior_1'] == 'Standing') & (filtered_movedata['Task'] == 'Task stand') |
    (filtered_movedata['Behavior_1'] == 'Shaking') |
    (filtered_movedata['Behavior_1'] == 'Galloping')
)

filtered_movedata = filtered_movedata[filter_conditions]

# 필터링된 데이터에서 Behavior_1의 상태 빈도 계산 및 출력
behavior_counts = filtered_movedata['Behavior_1'].value_counts()
print("Filtered Behavior Counts:")
print(behavior_counts)

# 5. 필요한 컬럼만 선택
sensor_columns = ['GNeck_x', 'GNeck_y', 'GNeck_z','ANeck_x', 'ANeck_y', 'ANeck_z']
activity_column = 'Behavior_1'

# 선택한 컬럼으로 새로운 데이터셋 생성 (복사본 명시적으로 생성)
sensor_data = filtered_movedata[sensor_columns].copy()
activity_data = filtered_movedata[activity_column].copy()

# 6. 결측치 처리 (필요 시)
sensor_data.fillna(0, inplace=True)
activity_data.fillna('undefined', inplace=True)

# 7. 라벨 인코딩 (활동 상태를 숫자로 변환)
le = LabelEncoder()
activity_data_encoded = le.fit_transform(activity_data)

# 8. 데이터셋을 훈련 및 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(sensor_data, activity_data_encoded, test_size=0.2, random_state=42)

# 9. 레이블을 원-핫 인코딩 (클래스 수 6개로 고정)
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

# 10. 데이터 차원 맞추기 (CNN을 위한 3차원으로 변환)
X_train_reshaped = np.expand_dims(X_train.values, axis=-1)
X_test_reshaped = np.expand_dims(X_test.values, axis=-1)

# 11. 모델 설계 (CNN + LSTM 구조)
model = Sequential()

# CNN 레이어 추가
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# LSTM 레이어 추가
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 클래스 수 6개로 고정

# 12. 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 13. 모델 훈련
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# 14. 모델 저장
model.save('pet_activity_cnn_rnn_model.h5')
