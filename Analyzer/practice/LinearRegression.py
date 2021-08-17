import numpy as np
import pandas as pd
from sklearn import linear_model

dir = 'data_regression/'
# DF = pd.read_csv(dir+'エゾ鹿.csv')
# y = DF['推定生息数']
# x1 = DF['前年の推定生息数']
# x2 = DF['前年の捕獲数']
# x = x1 - x2
# x = np.c_[np.array(x1).reshape(-1,1), np.array(x2).reshape(-1,1)]

# DF = pd.read_csv(dir+'経済指標.csv')
# y = DF['県内総生産額(千億円)']
# x1 = DF['県民所得(千億円)']
# x2 = DF['企業所得(千億円)']
# x = np.c_[np.array(x1).reshape(-1,1), np.array(x2).reshape(-1,1)]

DF = pd.read_csv(dir+'賃貸住宅.csv')
y = DF['賃料(万円)']
x1 = DF['面積(平方メートル)']
x2 = DF['築年(年)']
x3 = DF['ペット可']
x = np.c_[np.array(x1).reshape(-1,1), np.array(x2).reshape(-1,1), np.array(x3).reshape(-1,1)]


# x = np.array(x).reshape(-1,1)


def scratch():
    Sxx = np.sum(x*x)-np.sum(x)*np.sum(x)/x.size
    Sxy = np.sum(x*y)-np.sum(x)*np.sum(y)/x.size
    beta1_hat = Sxy/Sxx
    beta0_hat = np.mean(y) - beta1_hat*np.mean(x)
    
    Syy = np.sum(y*y) - np.sum(y)*np.sum(y)/y.size
    kiyoritu = Sxy*Sxy /Sxx/Syy
    
    y_hat = beta0_hat + beta1_hat*x
    zansa = y - y_hat
    MAE = np.mean(np.abs(zansa))

    print(f'切片β0の推定値:{beta0_hat}')
    print(f'回帰係数β1の推定値:{beta1_hat}')
    print(f'寄与率:{kiyoritu}')
    print(f'MAE:{MAE}')
    

def library():
    model = linear_model.LinearRegression()
    model.fit(x,y)
        
    beta1_hat = model.coef_
    beta0_hat = model.intercept_
    kiyoritu = model.score(x,y)
    y_hat = model.predict(x)
    zansa = y - y_hat
    MAE = np.mean(np.abs(zansa))
    
    print(f'切片β0の推定値:{beta0_hat}')
    print(f'回帰係数β1の推定値:{beta1_hat}')
    print(f'寄与率:{kiyoritu}')
    print(f'MAE:{MAE}')

if __name__ == '__main__':
    # scratch()
    # print()
    library()