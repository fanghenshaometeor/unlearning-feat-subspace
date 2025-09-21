import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model, args):
    if data_loader is None:
        return torch.zeros([0, args.num_classes]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            with torch.no_grad():
                if args.unlearn == "subspace":
                    output = model.forward_U(data)
                else:
                    output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    # print(n_target_train)
    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)


def get_SVC_MIA(shadow_train, shadow_test, target_train, target_test, model, args):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model, args)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model, args)

    target_train_prob, target_train_labels = collect_prob(target_train, model, args)
    target_test_prob, target_test_labels = collect_prob(target_test, model, args)

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    # shadow_train_entr = entropy(shadow_train_prob)
    # shadow_test_entr = entropy(shadow_test_prob)

    # target_train_entr = entropy(target_train_prob)
    # target_test_entr = entropy(target_test_prob)

    # shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    # shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)

    # if target_train is not None:
    #     target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    # else:
    #     target_train_m_entr = target_train_entr
    
    # if target_test is not None:
    #     target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    # else:
    #     target_test_m_entr = target_test_entr

    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    # acc_entr = SVC_fit_predict(
    #     shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    # )
    # acc_m_entr = SVC_fit_predict(
    #     shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    # )
    acc_entr = None
    acc_m_entr = None

    m = {
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
    }
    print(m)
    return m
