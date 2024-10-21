import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file customer_satisfaction.csv
df = pd.read_csv('restaurant_customer_satisfaction.csv')

# Hiển thị thông tin dữ liệu
print("Dữ liệu gốc:")
print(df.head())

# Chia dữ liệu thành đặc trưng và nhãn
X = df.drop('HighSatisfaction', axis=1)  # Các đặc trưng
y = df['HighSatisfaction']  # Nhãn: Mức độ hài lòng cao

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mã hóa các cột phân loại
categorical_columns = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit', 'DiningOccasion', 'MealType', 'OnlineReservation', 'DeliveryOrder', 'LoyaltyProgramMember']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le  # Lưu LabelEncoder cho từng cột

# In dữ liệu sau khi mã hóa
print("Dữ liệu sau khi mã hóa:")
print(X_train.head())

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f'Random Forest - Accuracy: {accuracy_rf}')
print(f'Random Forest - F1-score: {f1_rf}')

# Huấn luyện mô hình SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình SVM
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print(f'SVM - Accuracy: {accuracy_svm}')
print(f'SVM - F1-score: {f1_svm}')

# So sánh kết quả
if accuracy_rf > accuracy_svm:
    print("Mô hình Random Forest có độ chính xác cao hơn.")
else:
    print("Mô hình SVM có độ chính xác cao hơn.")

if f1_rf > f1_svm:
    print("Mô hình Random Forest có F1-score cao hơn.")
else:
    print("Mô hình SVM có F1-score cao hơn.")

# Tạo biểu đồ so sánh
labels = ['Random Forest', 'SVM']
accuracy_scores = [accuracy_rf, accuracy_svm]
f1_scores = [f1_rf, f1_svm]

x = range(len(labels))

fig, ax = plt.subplots(figsize=(10, 5))

# Vẽ biểu đồ độ chính xác
ax.bar(x, accuracy_scores, width=0.4, label='Độ chính xác', color='b', align='center')
# Vẽ biểu đồ F1-score
ax.bar([p + 0.4 for p in x], f1_scores, width=0.4, label='F1-score', color='r', align='center')

# Thiết lập trục và tiêu đề
ax.set_xticks([p + 0.2 for p in x])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.set_ylabel('Giá trị')
ax.set_title('So sánh độ chính xác và F1-score của các mô hình')
ax.legend()

# Hiển thị biểu đồ
plt.show()
