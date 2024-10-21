import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Đọc dữ liệu từ file customer_satisfaction.csv
df = pd.read_csv('restaurant_customer_satisfaction.csv')

# Hiển thị thông tin dữ liệu
st.title("Dự đoán Sự Hài Lòng Khách Hàng")
st.write("Dùng mô hình để dự đoán mức độ hài lòng của khách hàng dựa trên thông tin dịch vụ.")

# Chia dữ liệu thành đặc trưng và nhãn
X = df.drop('HighSatisfaction', axis=1)  # Các đặc trưng
y = df['HighSatisfaction']  # Nhãn: Mức độ hài lòng cao

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mã hóa các cột phân loại
categorical_columns = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit', 'DiningOccasion', 
                       'MealType', 'OnlineReservation', 'DeliveryOrder', 'LoyaltyProgramMember']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le  # Lưu LabelEncoder cho từng cột

# Hiện ô nhập cho người dùng
st.sidebar.header("Nhập thông tin khách hàng")
input_data = {}
for column in categorical_columns:
    input_data[column] = st.sidebar.selectbox(column, df[column].unique())

# Mã hóa dữ liệu đầu vào của người dùng
input_df = pd.DataFrame([input_data])
for column in categorical_columns:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Huấn luyện mô hình SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Dự đoán cho dữ liệu đầu vào của người dùng
rf_prediction = rf_model.predict(input_df)
svm_prediction = svm_model.predict(input_df)

# Hiển thị kết quả
st.subheader("Kết quả dự đoán")
st.write(f"**Dự đoán của Random Forest**: {rf_prediction[0]}")
st.write(f"**Dự đoán của SVM**: {svm_prediction[0]}")

# So sánh kết quả
# Giải thích sự khác biệt và chọn mô hình tối ưu
if accuracy_rf > accuracy_svm and f1_rf > f1_svm:
    st.write("Mô hình Random Forest có độ chính xác và F1-score cao hơn. Do đó, Random Forest là mô hình tối ưu hơn.")
    st.write("Random Forest có ưu điểm xử lý tốt dữ liệu phân loại, ít nhạy với nhiễu và không cần chuẩn hóa dữ liệu, do đó thích hợp hơn cho bài toán này.")
elif accuracy_svm > accuracy_rf and f1_svm > f1_rf:
    st.write("Mô hình SVM có độ chính xác và F1-score cao hơn. Do đó, SVM là mô hình tối ưu hơn.")
    st.write("SVM có thể tốt hơn khi dữ liệu được chuẩn hóa và có biên quyết định rõ ràng giữa các lớp.")
else:
    st.write("Cả hai mô hình đều có kết quả tương đương, tùy thuộc vào mục đích sử dụng mà có thể chọn mô hình.")
