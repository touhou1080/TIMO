from utils import *

# ------------------------------------------ Training Free ------------------------------------------


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_cache_keys = cache_keys

    affinity = val_features @ best_cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    print(
        "**** Cache val accuracy: {:.2f}. ****\n".format(cls_acc(cache_logits, val_labels)))

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        cfg, best_cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ best_cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)
                    ).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    return acc


def Refinement(cfg, new_cache_keys, cache_values, clip_weights, new_clip_weights,
               val_features, new_val_features, val_labels,
               test_features, new_test_features, test_labels):

    # Zero-shot CLIP
    R_fW = 100. * test_features @ clip_weights
    acc = cls_acc(R_fW, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']

    # Calculate R_f'F'
    R_fF = new_test_features @ new_cache_keys.t()

    # Calculate R_F'W'
    key_logits = new_cache_keys @ new_clip_weights
    key_logits = key_logits.softmax(1)
    cache_div = torch.sum(
        cache_values * torch.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]
    R_FW = (cache_div * gamma).exp()
    soft_cache_values = cache_values * R_FW

    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values

    ape_logits = R_fW + cache_logits * alpha
    acc = cls_acc(ape_logits, test_labels)
    print("**** Before search, test accuracy: {:.2f}. ****\n".format(acc))

    best_search_acc = 0
    R_fF = new_val_features @ new_cache_keys.t()
    R_fW = 100. * val_features @ clip_weights
    best_beta, best_alpha, best_gamma = 0, 0, 0
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step']
                 [0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step']
                  [1] + 0.1 for i in range(cfg['search_step'][1])]
    gamma_list = [i * cfg['search_scale'][2] / cfg['search_step'][2]
                  for i in range(cfg['search_step'][2])]
    for beta in beta_list:
        for alpha in alpha_list:
            for gamma in gamma_list:
                with torch.no_grad():
                    soft_cache_values = cache_values * \
                        (cache_div * gamma).exp()
                    cache_logits = ((-1) * (beta - beta * R_fF)
                                    ).exp() @ soft_cache_values
                    ape_logits = R_fW + cache_logits * alpha
                acc = cls_acc(ape_logits, val_labels)
                if acc > best_search_acc:
                    print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(
                        alpha, beta, gamma, acc))
                    best_search_acc = acc
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
    print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(
        best_search_acc))

    R_fW = 100. * test_features @ clip_weights
    R_fF = new_test_features @ new_cache_keys.t()

    soft_cache_values = cache_values * (cache_div * best_gamma).exp()
    cache_logits = ((-1) * (best_beta - best_beta * R_fF)
                    ).exp() @ soft_cache_values

    ape_logits = R_fW + cache_logits * best_alpha
    acc = cls_acc(ape_logits, test_labels)
    print("**** APE's test accuracy: {:.2f}. ****\n".format(acc))

    return acc


def APE(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, early_stop=False):

    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys = cache_keys.t().reshape(
        cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).cuda()

    cache_keys, cache_values = cache_keys.reshape(
        -1, feat_dim), cache_values.reshape(-1, cate_num)

    cfg['w'] = cfg['w_training_free']
    indices = cal_criterion(cfg, clip_weights, cache_keys,
                            only_use_txt=False, force=False)

    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]

    new_clip_weights = new_clip_weights / \
        new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features / \
        new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features / \
        new_val_features.norm(dim=-1, keepdim=True)

    if early_stop:
        return new_clip_weights, new_cache_keys, new_val_features, new_test_features

    acc = Refinement(cfg, new_cache_keys, cache_values, clip_weights, new_clip_weights,
                     val_features, new_val_features, val_labels,
                     test_features, new_test_features, test_labels)

    return acc


def GDA(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False):
    # normal distribution
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True)
                    for i in range(clip_weights.shape[1])])

    # KS Estimator.
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0)
                            for i in range(clip_weights.shape[1])])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov(
    ) + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())

    ps = torch.ones(clip_weights.shape[1]).cuda() * 1. / clip_weights.shape[1]
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

    # Evaluate
    # Grid search for hyper-parameter alpha
    best_val_acc = 0
    best_alpha = 0.1
    # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    for alpha in [10**i for i in range(-4, 5)]:
        if alpha_shift:
            val_logits = alpha * val_features.float() @ clip_weights.float() + \
                val_features.float() @ W + b
        else:
            val_logits = 100. * val_features.float() @ clip_weights.float() + alpha * \
                (val_features.float() @ W + b)

        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = alpha
    ############################################################################
    # mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])]).float()
    # center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(clip_weights.shape[1])])
    # cov = center_vecs.T.cov()
    # cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * cov + cov.trace() * torch.eye(center_vecs.shape[1]).cuda())

    # ps = torch.ones(clip_weights.shape[1]).cuda() * 1. / clip_weights.shape[1]
    # W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    # b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

    # # Evaluate
    # # Grid search for hyper-parameter alpha
    # best_val_acc = 0
    # best_alpha = 0.1
    # for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]:
    #     val_logits = alpha * val_features.float() @ clip_weights.float() + \
    #         val_features.float() @ W + b

    #     acc = cls_acc(val_logits, val_labels)
    #     if acc > best_val_acc:
    #         best_val_acc = acc
    #         best_alpha = alpha

    print("best_val_alpha: %s \t best_val_acc: %s" %
          (best_alpha, best_val_acc))
    alpha = best_alpha
    return alpha, W, b, best_val_acc


def GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights):

    # Parameter Estimation.
    with torch.no_grad():
        # Ours
        vecs = torch.load(
            cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
        labels = torch.load(
            cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        alpha, W, b, _ = GDA(vecs, labels, clip_weights,
                             val_features, val_labels)

        test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * \
            (test_features.float() @ W + b)
        notune_acc = cls_acc(test_logits, test_labels)
        print("training-free acc:", notune_acc)
    return notune_acc


def TIMO(cfg, val_features, val_labels, test_features, test_labels,
         clip_weights, clip_weights_all, matching_score, vecs_labels=None,
         grid_search=False, n_quick_search=-1, is_print=False):

    best_val_acc = 0
    best_alpha = 0.1

    with torch.no_grad():
        # Image Vecs
        if vecs_labels is None:
            vecs_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
            labels_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        else:
            vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

        vecs_t = clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        cate_num, prompt_num, _, = vecs_t.shape
        vecs_c, labels_c = vecs_v, labels_v

        if grid_search:
            if n_quick_search != -1:
                beta_list = [int(t) for t in torch.linspace(
                    1, prompt_num*2, n_quick_search)]
            else:
                beta_list = range(1, prompt_num*2)
        else:
            beta_list = [prompt_num]

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]  # c, s, d
            sliced_weights = weights.repeat(1, 2)[:, :beta]  # c, s

            # weight for instance based transfer
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)

            sliced_vecs_t = sliced_vecs_t.reshape(cate_num*beta, -1)
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance based transfer
            vecs_c = torch.cat([sliced_vecs_t, vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            alpha, W, b, val_acc = GDA(
                vecs_c, labels_c, clip_weights, val_features, val_labels, alpha_shift=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        alpha = best_alpha
        test_logits = alpha * test_features.float() @ clip_weights.float() + \
            (test_features.float() @ best_weights[0] + best_weights[1])
        acc = cls_acc(test_logits, test_labels)

        if is_print:
            print("best_val_alpha: %s \t best_val_acc: %s" %
                  (best_alpha, best_val_acc))
            print("best_beta:", best_beta)
            print("training-free acc:", acc)
            print()

    return acc

def timo_with_ape(cfg,cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,clip_weights_all,image_weights_all):
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys = cache_keys.t().reshape(
        cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).cuda()

    cache_keys, cache_values = cache_keys.reshape(
        -1, feat_dim), cache_values.reshape(-1, cate_num)

    cfg['w'] = cfg['w_training_free']
    indices = cal_criterion(cfg, clip_weights, cache_keys,
                            only_use_txt=False, force=False)

    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]

    new_clip_weights = new_clip_weights / \
        new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features / \
        new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features / \
        new_val_features.norm(dim=-1, keepdim=True)
    
    new_clip_weights_all = clip_weights_all[:,:,indices]
    new_clip_weights_all = new_clip_weights_all / new_clip_weights_all.norm(dim=-1, keepdim=True)
    new_image_weights_all = image_weights_all[:,:,indices]
    new_image_weights = new_image_weights_all.mean(dim=1)
    new_image_weights = new_image_weights / new_image_weights.norm(dim=1, keepdim=True)

    clip_weights_IGT, matching_score = image_guide_text(cfg, new_clip_weights_all, new_image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    clip_weights_IGT, matching_score = image_guide_text_search(cfg, new_clip_weights_all, new_val_features, val_labels, new_image_weights)
    #acc_free = TIMO(cfg, new_val_features, val_labels, new_test_features, test_labels, 
    #    clip_weights_IGT, new_clip_weights_all, matching_score,vecs_labels=None,
    #    grid_search=True, n_quick_search=10, is_print=True)
    vecs_labels=None
    n_quick_search=10
    is_print=True
    grid_search=True

    best_val_acc = 0
    best_alpha = 0.1
    with torch.no_grad():
        # Image Vecs
        if vecs_labels is None:
            vecs_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
            labels_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        else:
            vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

        new_vecs_v = vecs_v[:,indices]
        vecs_t = new_clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        cate_num, prompt_num, _, = vecs_t.shape
        vecs_c, labels_c = new_vecs_v, labels_v

        if grid_search:
            if n_quick_search != -1:
                beta_list = [int(t) for t in torch.linspace(
                    1, prompt_num*2, n_quick_search)]
            else:
                beta_list = range(1, prompt_num*2)
        else:
            beta_list = [prompt_num]

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]  # c, s, d
            sliced_weights = weights.repeat(1, 2)[:, :beta]  # c, s

            # weight for instance based transfer
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)

            sliced_vecs_t = sliced_vecs_t.reshape(cate_num*beta, -1)
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            #print(f"Debug before error:")
            #print(f"tmp shape: {tmp.shape}, dtype: {tmp.dtype}, device: {tmp.device}")
            #print(f"tmp values: {tmp.flatten()[:10]}")  # 打印前10个值
            #print(f"sliced_vecs_t shape: {sliced_vecs_t.shape}, device: {sliced_vecs_t.device}")
            #print(f"new_vecs_v shape: {new_vecs_v.shape}, device: {new_vecs_v.device}")
            #print(f"labels_v shape: {labels_v.shape}, dtype: {labels_v.dtype}, device: {labels_v.device}")
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance based transfer
            vecs_c = torch.cat([sliced_vecs_t, new_vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            alpha, W, b, val_acc = GDA(
                vecs_c, labels_c, clip_weights_IGT, new_val_features, val_labels, alpha_shift=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        alpha = best_alpha
        test_logits = alpha * new_test_features.float() @ clip_weights_IGT.float() + \
            (new_test_features.float() @ best_weights[0] + best_weights[1])
        acc = cls_acc(test_logits, test_labels)

        if is_print:
            print("best_val_alpha: %s \t best_val_acc: %s" %
                  (best_alpha, best_val_acc))
            print("best_beta:", best_beta)
            print("training-free acc:", acc)
            print()
    return acc

def _timo_grid_search_eval(cfg, val_features, val_labels, test_features, test_labels,
                           clip_weights_IGT, clip_weights_all, matching_score, indices):
    """
    Helper function to perform TIMO's grid search on a feature subspace and return
    both test and validation accuracy.
    """
    best_val_acc = 0
    best_alpha = 0.1
    best_beta = -1
    best_weights = None
    
    with torch.no_grad():
        # Load image vectors (from cache) and project to subspace
        vecs_v = torch.load(cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
        labels_v = torch.load(cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        subspace_vecs_v = vecs_v[:, indices]

        # Prepare text vectors
        vecs_t = clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        cate_num, prompt_num, _, = vecs_t.shape

        # Define search space for beta
        beta_list = [int(t) for t in torch.linspace(1, prompt_num * 2, 10)] # n_quick_search=10

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]
            sliced_weights = weights.repeat(1, 2)[:, :beta]
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)
            sliced_vecs_t = sliced_vecs_t.reshape(cate_num * beta, -1)
            
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance-based transfer
            vecs_c = torch.cat([sliced_vecs_t, subspace_vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            # GDA for finding alpha
            alpha, W, b, val_acc = GDA(vecs_c, labels_c, clip_weights_IGT, val_features, val_labels, alpha_shift=True)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        # Evaluate test accuracy with the best found hyperparameters
        test_logits = best_alpha * test_features.float() @ clip_weights_IGT.float() + \
                      (test_features.float() @ best_weights[0] + best_weights[1])
        test_acc = cls_acc(test_logits, test_labels)

        print(f"Search Results -> best_beta: {best_beta}, best_alpha: {best_alpha:.4f}, val_acc: {best_val_acc:.4f}, test_acc: {test_acc:.4f}")

    return test_acc, best_val_acc

def timo_with_ape_v3(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_weights_all, image_weights_all):
    """
    Performs a search over the number of feature dimensions (k) to find the optimal
    subspace for TIMO, effectively combining APE (feature selection) and TIMO.
    The best k is selected based on validation set performance.
    """
    # 1. 定义要搜索的特征数量 k 的列表
    feat_dim = clip_weights.shape[0]
    # 使用0.1到1.0的比例来定义k值，覆盖从10%到100%的特征维度
    k_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_list = sorted(list(set([int(feat_dim * r) for r in k_ratios])))
    
    print(f"\nSearching for the best feature dimension 'k' in: {k_list}")

    # 2. 在循环外计算一次特征重要性分数 (APE的核心)
    # 我们将 training_free 设置为 False 并传入一个大的 k 值来获取完整的排序标准
    cfg_copy = cfg.copy()
    cfg_copy['training_free_feat_num'] = feat_dim # 获取所有维度的分数
    criterion_scores = cal_criterion_scores(cfg, clip_weights, cache_keys, only_use_txt=True)


    best_k = -1
    best_val_acc_for_k = -1.0
    final_test_acc = -1.0
    
    # 3. 遍历 k_list 进行搜索
    for k in k_list:
        print(f"\n--- Testing with k = {k} ---")
        
        # --- APE 部分: 特征选择 ---
        # 从预先计算的分数中获取 top-k 索引
        _, indices = torch.topk(criterion_scores, k=k)

        # --- 子空间投影 ---
        subspace_val_features = val_features[:, indices]
        subspace_test_features = test_features[:, indices]
        subspace_clip_weights_all = clip_weights_all[:, :, indices]
        subspace_image_weights_all = image_weights_all[:, :, indices]
        
        # --- TIMO 部分: 在子空间中运行 ---
        # IGT (Image-Guided Text)
        # 注意：这里我们直接使用子空间中的特征进行TIMO的后续步骤
        subspace_image_weights = subspace_image_weights_all.mean(dim=1)
        subspace_image_weights = subspace_image_weights / subspace_image_weights.norm(dim=1, keepdim=True)
        
        # image_guide_text_search 内部会处理归一化并搜索最佳 gamma
        clip_weights_IGT, matching_score = image_guide_text_search(cfg, subspace_clip_weights_all, subspace_val_features, val_labels, subspace_image_weights)
        
        # 调用辅助函数，在子空间上执行TIMO的alpha/beta网格搜索
        # 这个函数会返回当前k值对应的测试集和验证集准确率
        current_test_acc, current_val_acc = _timo_grid_search_eval(
            cfg, 
            subspace_val_features, val_labels, 
            subspace_test_features, test_labels,
            clip_weights_IGT, # 注意转置
            subspace_clip_weights_all, 
            matching_score, 
            indices
        )
        
        # 4. 根据验证集准确率，更新最佳 k 和对应的测试集准确率
        if current_val_acc > best_val_acc_for_k:
            best_val_acc_for_k = current_val_acc
            final_test_acc = current_test_acc
            best_k = k
            print(f"Found new best k: {best_k} (Val Acc: {best_val_acc_for_k:.4f}, Test Acc: {final_test_acc:.4f})")

    print("\n--------------------------------------------------")
    print(f"TIMO_APE Final Result:")
    print(f"Best feature dimension (k): {best_k}")
    print(f"Best validation accuracy: {best_val_acc_for_k:.4f}")
    print(f"Test accuracy with best k: {final_test_acc:.4f}")
    print("--------------------------------------------------")
    
    return final_test_acc
