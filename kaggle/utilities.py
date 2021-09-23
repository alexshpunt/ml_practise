from matplotlib.pyplot import * 

def measure_model(model, test_data, target_test_data, color='g'):
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
    pred = model.predict_proba(test_data)[:,1]
    acc = accuracy_score(target_test_data, pred > 0.5)
    auc = roc_auc_score(target_test_data, pred)
    fpr, tpr, thr = roc_curve(target_test_data, pred)
    plot(fpr, tpr, color)
    xlabel('false positive')
    ylabel('true positive')
    return {
        'acc':acc,
        'auc':auc,
    }