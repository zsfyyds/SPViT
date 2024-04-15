import numpy as np

class ARIMA:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q


    def difference(self, data, interval):
        return [data[i] - data[i - interval] for i in range(interval, len(data))]


    def inverse_difference(self, history, yhat, interval):
        return yhat + history[-interval]


    def fit(self, data):
        self.data = data
        self.history = [x for x in data]
        self.residuals = []
        return self


    def forecast(self):
        predictions = []
        for t in range(len(self.data)):
            model = self._ARIMA()
            yhat = model['coef'][0]
            obs = self.data[t]
            predictions.append(yhat)
            error = obs - yhat
            self.residuals.append(error)
        return predictions


    def _ARIMA(self):
        history = self.history
        model = {'coef': [0.0 for _ in range(self.p)]}
        residuals = self.residuals
        data = self.data
        p = self.p
        q = self.q
        d = self.d
        for t in range(self.p, len(history)):
            model['coef'].append(history[t] - history[t - 1])
        return model


