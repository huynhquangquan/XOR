import numpy as np
import matplotlib.pyplot as plt

errors = []

# Bước 1: Chuẩn bị dữ liệu
# Dữ liệu mẫu
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Dữ liệu đầu vào và đầu ra cho bài toán XOR
input_neurons = 2  # Số neuron đầu vào (2 đặc trưng x1 và x2)
hidden_neurons = 2  # Số neuron trong lớp ẩn
output_neurons = 1  # Số neuron đầu ra (1 đầu ra)

# Bước 2: Trọng số và hàm kích hoạt
# Hàm kích hoạt sigmoid và đạo hàm của nó
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Khởi tạo trọng số ngẫu nhiên
np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Bước 3: Tính toán truyền tiến
# Chức năng tính toán truyền tiến (Forward Pass Function)
def forward_pass(X):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return hidden_layer_output, predicted_output

# Bước 4: Backpropagation và cập nhật trọng số
# Chức năng lan truyền ngược (Backpropagation Function)
def backpropagation(hidden_layer_output, predicted_output):
    error = y - predicted_output
    mean_squared_error = np.mean(np.square(error))
    errors.append(mean_squared_error)

    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    return d_predicted_output, d_hidden_layer


# Chức năng cập nhật trọng số (Update Weight Function)
def update_weights(hidden_layer_output, d_predicted_output, d_hidden_layer):
    global weights_input_hidden, weights_hidden_output  # Sử dụng biến toàn cục

    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Bước 5: Huấn luyện mạng
# Huấn luyện mạng MLP
learning_rate = 0.5
epochs = 10000
for epoch in range(epochs):
    # Tính toán truyền tiến
    hidden_layer_output, predicted_output = forward_pass(X)

    # Lan truyền ngược
    d_predicted_output, d_hidden_layer = backpropagation(hidden_layer_output, predicted_output)

    # Cập nhật trọng số
    update_weights(hidden_layer_output, d_predicted_output, d_hidden_layer)

# Kiểm tra kết quả sau khi huấn luyện
print("Output sau khi huấn luyện:\n",predicted_output.round())
print("Dự đoán chính xác:\n", predicted_output)

# BONUS: Vẽ biểu đồ plot
# Vẽ biểu đồ sai số qua từng vòng lặp
plt.plot(errors)
plt.xlabel("Vòng Lặp")
plt.ylabel("Sai Số")
plt.title("Sai Số qua các Vòng Lặp")
plt.show()

# Lưới tọa độ để vẽ mặt phẳng phân loại
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Truyền tiến cho từng điểm trên lưới để vẽ mặt phẳng phân loại
hidden_layer_input_grid = np.dot(grid_points, weights_input_hidden)
hidden_layer_output_grid = sigmoid(hidden_layer_input_grid)
output_layer_input_grid = np.dot(hidden_layer_output_grid, weights_hidden_output)
predicted_output_grid = sigmoid(output_layer_input_grid)
predicted_output_grid = predicted_output_grid.reshape(xx.shape)

# Vẽ mặt phẳng phân loại
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, predicted_output_grid, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.8)
plt.colorbar(label="Đầu ra Dự đoán")

# Vẽ các điểm huấn luyện
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=100, edgecolor='k', cmap="coolwarm", label="Điểm Huấn Luyện")
plt.title("Mặt Phẳng Phân Loại của Bài Toán XOR")
plt.xlabel("Đầu Vào 1")
plt.ylabel("Đầu Vào 2")
plt.legend()
plt.grid(True)
plt.show()
