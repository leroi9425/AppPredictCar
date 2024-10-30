import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("btvn/AppPredictCar/data.csv")
pd.set_option('display.max_columns', None)
df.head()

# xoa cot thua
df.drop(['car_ID','symboling'],axis=1,inplace=True)
df.head()

# chi lay hang k lay model
df['CarName'] = df['CarName'].apply(lambda x: x.split(" ")[0])

# thay the hang xe
def replace_name(x,y):
    df['CarName'].replace(x,y,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

# chuyen data sang dang so
df['doornumber'] = df['doornumber'].replace({'four': 4, 'two': 2}).astype('int64')

df['cylindernumber'] = df['cylindernumber'].replace({'four': 4, 'six': 6, 'five': 5, 'eight': 8, 'two': 2, 'three': 3, 'twelve': 12}).astype('int64')

# loai bo ngoai le
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
old_df = df_no_outliers

# gan X Y
categorical_column = ['fueltype','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem','CarName'] 
df_no_outliers = pd.get_dummies(df_no_outliers, columns=categorical_column, drop_first=True)

X = df_no_outliers.drop('price', axis=1).astype(float).values
y = df_no_outliers['price'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)


# linear
ones_train = np.ones((X_train.shape[0], 1))
ones_test = np.ones((X_test.shape[0], 1))
Xbar_train = np.concatenate((ones_train, X_train), axis=1)
Xbar_test = np.concatenate((ones_test, X_test), axis=1)

A = np.dot(Xbar_train.T, Xbar_train)
b = np.dot(Xbar_train.T, y_train)
w = np.dot(np.linalg.pinv(A), b)

# tree
class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2):
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        num_samples, num_features = X.shape
        # Kiểm tra điều kiện dừng
        if num_samples < self.min_samples_split or len(set(y)) == 1:
            return np.mean(y)  # Trả về giá trị trung bình

        # Tìm thuộc tính và ngưỡng tối ưu
        best_feature, best_threshold = self._best_split(X, y)

        # Nếu không tìm thấy thuộc tính nào tốt, trả về giá trị trung bình
        if best_feature is None:
            return np.mean(y)

        # Tách dữ liệu thành các nhánh
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices])
        right_tree = self._build_tree(X[right_indices], y[right_indices])

        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                mse = self._calculate_mse(y, left_indices, right_indices)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, y, left_indices, right_indices):
        left_mean = np.mean(y[left_indices]) if len(y[left_indices]) > 0 else 0
        right_mean = np.mean(y[right_indices]) if len(y[right_indices]) > 0 else 0
        mse = (np.sum((y[left_indices] - left_mean) ** 2) +
               np.sum((y[right_indices] - right_mean) ** 2)) / len(y)
        return mse

    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _predict(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree  # Giá trị dự đoán (trung bình)

        feature, threshold, left_tree, right_tree = tree
        if sample[feature] < threshold:
            return self._predict(sample, left_tree)
        else:
            return self._predict(sample, right_tree)

# Tạo mô hình cây quyết định
model = DecisionTreeRegressor(min_samples_split=5)

# Huấn luyện mô hình
model.fit(X_train, y_train)
    

def run_app():
    from flask import Flask, render_template, request
    # app = Flask(__name__, static_url_path='/static')
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return render_template('index.html')
    @app.route('/result')
    def result():
        return render_template('index.html')

    @app.route('/input', methods=['POST'])
    def predByLinear():

        CarName = request.form['CarName']
        justname = request.form['justname']
        fueltype = request.form['fueltype']
        aspiration = request.form['aspiration']
        doornumber = int(request.form['doornumber'])
        carbody = request.form['carbody']
        drivewheel = request.form['drivewheel']
        enginelocation = request.form['enginelocation']
        wheelbase = float(request.form['wheelbase'])
        carlength = float(request.form['carlength'])
        carwidth = float(request.form['carwidth'])
        carheight = float(request.form['carheight'])
        curbweight = float(request.form['curbweight'])
        enginetype = request.form['enginetype']
        cylindernumber = int(request.form['cylindernumber'])
        enginesize = float(request.form['enginesize'])
        fuelsystem = request.form['fuelsystem']
        boreratio = float(request.form['boreratio'])
        stroke = float(request.form['stroke'])
        compressionratio = float(request.form['compressionratio'])
        horsepower = float(request.form['horsepower'])
        peakrpm = float(request.form['peakrpm'])
        citympg = float(request.form['citympg'])
        highwaympg = float(request.form['highwaympg'])
        
        row = np.array([[
                        CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, wheelbase, carlength,
                        carwidth, carheight, curbweight, enginetype, cylindernumber, enginesize, fuelsystem,boreratio,
                        stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg
                         ]])

        car = pd.DataFrame({
                                'CarName': [CarName],
                                'fueltype': [fueltype],
                                'aspiration': [aspiration],
                                'doornumber': [doornumber],
                                'carbody': [carbody],
                                'drivewheel': [drivewheel],
                                'enginelocation': [enginelocation],
                                'wheelbase': [wheelbase],
                                'carlength': [carlength],
                                'carwidth': [carwidth],
                                'carheight': [carheight],
                                'curbweight': [curbweight],
                                'enginetype': [enginetype],
                                'cylindernumber': [cylindernumber],
                                'enginesize': [enginesize],
                                'fuelsystem': [fuelsystem],
                                'boreratio': [boreratio],
                                'stroke': [stroke],
                                'compressionratio': [compressionratio],
                                'horsepower': [horsepower],
                                'peakrpm': [peakrpm],
                                'citympg': [citympg],
                                'highwaympg': [highwaympg]
                            })

        # thu ghep data roi ma hoa sau do lai tach ra tiep
        new_car = pd.DataFrame(car)
        new_car_encoded = pd.get_dummies(new_car, columns=categorical_column, drop_first=False)
        df_no_outliers_encoded = df_no_outliers.drop(columns='price')
        new_car_encoded = new_car_encoded.reindex(columns=df_no_outliers_encoded.columns, fill_value=0)

        option = request.form['option']
        price_predicted = [0]
        result_price = 0

        if option == 'linear':
            new_car_encoded.insert(0, 'bias', 1)
            price_predicted = np.dot(new_car_encoded, w)
            result_price = price_predicted[0]
        else:
            a_new_car = new_car_encoded.astype(float).values
            result_price = model._predict(a_new_car[0], model.tree)

        #return render_template('index.html', ketqua = int(price_predicted[0]))
        return render_template('index.html', 
                                justname = justname,
                                carname = CarName,
                                fueltype = fueltype,
                                aspiration = aspiration,
                                doornumber = doornumber,
                                carbody = carbody,
                                drivewheel = drivewheel,
                                enginelocation = enginelocation,
                                wheelbase = wheelbase,
                                carlength = carlength,
                                carwidth = carwidth,
                                carheight = carheight,
                                curbweight = curbweight,
                                enginetype = enginetype,
                                cylindernumber = cylindernumber,
                                enginesize = enginesize,
                                fuelsystem = fuelsystem,
                                boreratio = boreratio,
                                stroke = stroke,
                                compressionratio = compressionratio,
                                horsepower = horsepower,
                                peakrpm = peakrpm,
                                citympg = citympg,
                                highwaympg = highwaympg,
                               ketqua = int(result_price))
    
    if __name__ == '__main__':
        app.run(debug=True)
        
def main():
    run_app()


main()