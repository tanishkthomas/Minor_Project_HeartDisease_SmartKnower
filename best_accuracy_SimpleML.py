class BestAccuracy(object):
    
    def _init_(self, mod, X, Y, i):
        self.i = i
        self.model = mod
        self.X = X
        self.Y = Y
        
    def Accuracy(self):
        t = self.i
        largest_accuracy = 0
        while t>0:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, random_state =t)
            
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
    
            model = self.model
        
            model.fit(X_train, Y_train)
        
            Y_pred = model.predict(X_test)
        
            r = metrics.accuracy_score(Y_test, Y_pred)*100
            print(r)
            if r>largest_accuracy:
                largest_accuracy = r
                state = t
            t -= 1
            
        print("State:- "+ str(state))
        print("Accuracy:- "+ str(largest_accuracy))