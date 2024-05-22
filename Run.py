import numpy as np
np.random.seed(0)
from Cuts import *
from tqdm import tqdm

def run(data_path,is_standard=False,train_size=121,test_size=100,multi_row=True):
    result_cgf1 = np.zeros((train_size, test_size))
    s = [(i/10, j/10) for i in range(11) for j in range(11)]
    
    if multi_row:
        result_cgf2 = np.zeros((train_size, test_size))
        result_cgf5 = np.zeros((train_size, test_size))
        result_cgf10 = np.zeros((train_size, test_size))

    mu2 = np.random.exponential(scale=1.0, size=(train_size, 2))
    mu2 = mu2 / mu2.sum(axis=1).reshape(-1, 1)
    mu5 = np.random.exponential(scale=1.0, size=(train_size, 5))
    mu5 = mu5 / mu5.sum(axis=1).reshape(-1, 1)
    mu10 = np.random.exponential(scale=1.0, size=(train_size, 10))
    mu10 = mu10 / mu10.sum(axis=1).reshape(-1, 1)

    with tqdm(range(train_size), desc=data_path) as pbar:
        for i in pbar:
            for j in range(test_size):
                data = np.load(f'data/{data_path}/train/{data_path}_train_{j}.npy', allow_pickle=True).item()
                A, c, b = data['A'], data['c'], data['b']
                cut_size = Cut_size(A, c, b, is_standard=is_standard)
                try:
                    result_cgf1[i, j] = cut_size.get_cgf1_treesize(s[i])
                    if multi_row:
                        result_cgf2[i, j] = cut_size.get_cgfk_treesize(mu2[i], 2)
                        result_cgf5[i, j] = cut_size.get_cgfk_treesize(mu5[i], 5)
                        result_cgf10[i, j] = cut_size.get_cgfk_treesize(mu10[i], 10)
                except Exception as e:
                    print(e)
                    result_cgf1[i, j] = np.nan
                    if multi_row:
                        result_cgf2[i, j] = np.nan
                        result_cgf5[i, j] = np.nan
                        result_cgf10[i, j] = np.nan
            if i >= 1:
                if multi_row:
                    pbar.set_postfix({
                        "GMI Average Treesize": f"{result_cgf1.mean(axis=1)[0]:.2f}",
                        "Best 1 row CGF Treesize": f"{result_cgf1.mean(axis=1)[1:i+1].min():.2f}",
                        "Best 2 rows CGF Treesize": f"{result_cgf2.mean(axis=1)[:i+1].min():.2f}",
                        "Best 5 rows CGF Treesize": f"{result_cgf5.mean(axis=1)[:i+1].min():.2f}",
                        "Best 10 rows CGF Treesize": f"{result_cgf10.mean(axis=1)[:i+1].min():.2f}"
                    })
                else:
                    pbar.set_postfix({
                        "GMI Average Treesize": f"{result_cgf1.mean(axis=1)[0]:.2f}",
                        "Best 1 row CGF Treesize": f"{result_cgf1.mean(axis=1)[1:i+1].min():.2f}"
                    })

    best_s = s[result_cgf1.mean(axis=1).argmin()]
    if multi_row:
        best_mu2 = mu2[result_cgf2.mean(axis=1).argmin()]
        best_mu5 = mu5[result_cgf5.mean(axis=1).argmin()]
        best_mu10 = mu10[result_cgf10.mean(axis=1).argmin()]

        training_result_dict = {
            'result_cgf1': result_cgf1,
            'result_cgf2': result_cgf2,
            'result_cgf5': result_cgf5,
            'result_cgf10': result_cgf10,
            'best_s': best_s,
            'best_mu2': best_mu2,
            'best_mu5': best_mu5,
            'best_mu10': best_mu10
        }
    else:
        training_result_dict = {
            'result_cgf1': result_cgf1,
            'best_s': best_s
        }

    np.save(f'data/{data_path}/training_data_dict.npy', training_result_dict)
    print("Training Completed")

    # Testing
    print("Testing Started")
    if multi_row:
        test_results = {
            'test_gmi': np.zeros(test_size),
            'test_cgf1': np.zeros(test_size),
            'test_cgf2': np.zeros(test_size),
            'test_cgf5': np.zeros(test_size),
            'test_cgf10': np.zeros(test_size)
        }
    else:
        test_results = {
            'test_gmi': np.zeros(test_size),
            'test_cgf1': np.zeros(test_size)
        }

    for i in tqdm(range(test_size)):
        data = np.load(f'data/{data_path}/test/{data_path}_test_{i}.npy', allow_pickle=True).item()
        A, c, b = data['A'], data['c'], data['b']
        cut_size = Cut_size(A, c, b, is_standard=is_standard)
        try:
            test_results['test_gmi'][i] = cut_size.get_cgf1_treesize([0.0, 0.0])
            test_results['test_cgf1'][i] = cut_size.get_cgf1_treesize(best_s)
            if multi_row:
                test_results['test_cgf2'][i] = cut_size.get_cgfk_treesize(best_mu2, 2)
                test_results['test_cgf5'][i] = cut_size.get_cgfk_treesize(best_mu5, 5)
                test_results['test_cgf10'][i] = cut_size.get_cgfk_treesize(best_mu10, 10)
        except Exception as e:
            print(e)
            test_results['test_gmi'][i] = np.nan
            test_results['test_cgf1'][i] = np.nan
            if multi_row:
                test_results['test_cgf2'][i] = np.nan
                test_results['test_cgf5'][i] = np.nan
                test_results['test_cgf10'][i] = np.nan
    test_results['test_gmi'] = test_results['test_gmi'].mean()
    test_results['test_cgf1'] = test_results['test_cgf1'].mean()
    if multi_row:
        test_results['test_cgf2'] = test_results['test_cgf2'].mean()
        test_results['test_cgf5'] = test_results['test_cgf5'].mean()
        test_results['test_cgf10'] = test_results['test_cgf10'].mean()

    np.save(f'data/{data_path}/test_data_dict.npy', test_results)
    print(test_results)

