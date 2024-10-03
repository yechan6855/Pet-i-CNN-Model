import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# 저장된 모델 불러오기
model = load_model('pet_activity_cnn_rnn_model4.h5')

# 입력 데이터 (자이로스코프 x, y, z 및 가속도계 x, y, z 순서)
input_data = np.array([[-0, -0, -0, -0, 0, -0]])  # 예시 데이터

# 입력 데이터 전처리 (CNN 모델을 위해 3차원 배열로 변환)
input_data_reshaped = np.expand_dims(input_data, axis=-1)  # (1, 6, 1)

# 모델로 예측
prediction = model.predict(input_data_reshaped)

# 예측 결과 해석 (가장 높은 확률을 가진 클래스 선택)
predicted_class = np.argmax(prediction, axis=-1)

# 라벨 인코더 불러오기
le = LabelEncoder()
activity_data = ['Walking', 'Trotting', 'Sitting', 'Standing', 'Shaking', 'Galloping']

# 라벨 인코더에 활동 상태 학습시키기
le.fit(activity_data)

# 예측된 클래스 출력
predicted_activity = le.inverse_transform(predicted_class)
print(f"예측된 활동 상태: {predicted_activity[0]}")