import numpy as np
from ldl_flc import LDL_FLC, fuzzy_cmeans, solve_LDM
from ldl_metrics import score
from util import load_dict
import csv
import sys


def run_and_save(dataset: str, out_csv: str):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    train_inds = load_dict(dataset, "train_inds")
    test_inds = load_dict(dataset, "test_inds")

    rows = [("cheby","clark","can","kl","cosine","inter")]
    for i in range(10):
        print('training ' + str(i + 1) + ' fold')
        train_x, train_y = X[train_inds[i]], Y[train_inds[i]]
        test_x, test_y = X[test_inds[i]], Y[test_inds[i]]

        l1 = 0.001; l2 = 0.01; g = 5
        U = fuzzy_cmeans(train_y, g)
        manifolds = solve_LDM(train_y, U)

        ldl_flc = LDL_FLC(g, l1, l2)
        ldl_flc.fit(train_x, train_y, U, manifolds)
        ldl_flc.solve()

        cheby, clark, can, kl, cosine, inter = score(test_y, ldl_flc.predict(test_x))
        rows.append((cheby, clark, can, kl, cosine, inter))

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print('Saved Python metrics to', out_csv)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else 'SJAFFE'
    out = sys.argv[2] if len(sys.argv) > 2 else ds + '/ldlfc_python_metrics.csv'
    run_and_save(ds, out)

