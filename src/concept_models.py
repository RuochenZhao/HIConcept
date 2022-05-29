# Concept Models
import torch.nn as nn
import torch.nn.functional as F
import torch

##THIS ONE
class topic_model_main(nn.Module):
    def __init__(self, criterion, classifier, f_train, n_concept, thres, device, args):
        super().__init__()
        # print(f_train.shape) #bs, 100, 391
        self.topic_vector = nn.Parameter(self.init_concept(f_train.shape[-1], n_concept), requires_grad=True) #hidden_dim, 10
        # print('self.topic_vector.shape: ', self.topic_vector.shape)
        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concept, args.hidden_dim), requires_grad = True)
        # if args.extra_layers>0:
        #     self.extra_recs = [nn.Parameter(self.init_concept(args.hidden_dim, args.hidden_dim), requires_grad = True).to(device) for i in range(args.extra_layers)]
        self.rec_vector_2 = nn.Parameter(self.init_concept(args.hidden_dim, f_train.shape[-1]), requires_grad = True)

        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.n_concept = n_concept
        self.thres = 0.3
        self.device = device
        self.bert = (args.model_name=='bert')
        self.args = args
        self.criterion = criterion


    def init_concept(self, embedding_dim, n_concept):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concept) + r_1
        return concept
    
    def flip_loss(self, y_pred, pred_perturbed, targets):
        if self.args.loss=='flip':
            # maxmize the mean
            # minimize the negative
            if self.bert:
                y_pred = F.softmax(y_pred, dim = 1)[:, 1]
                pred_perturbed = F.softmax(pred_perturbed, dim = 1)[:, 1]
                return -torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)))
            return -torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)))
        else:
            # print('here')
            loss = self.criterion(pred_perturbed.squeeze(), targets.squeeze().float()) 
            return - loss
        
    def concept_sim(self, topic_prob_n):
        # maximize the top k 
        # topic_prob_n : #32, 100, 4
        batch_size = topic_prob_n.shape[0]
        res = torch.reshape(topic_prob_n,(-1,self.n_concept))#3200, 4
        res = torch.transpose(res,0,1)#4, 3200
        res = torch.topk(res,k=batch_size//4,sorted=True).values#4, 32
        res = torch.mean(res)
        return - res
        # return 0

    def concept_far(self, topic_vector_n):
        # topic_vector_n: #hidden_dim, n_concept
        # after norm: n_concept, n_concept
        return torch.mean(torch.mm(torch.transpose(topic_vector_n, 0, 1), topic_vector_n) - torch.eye(self.n_concept).to(self.device))
        # # new way of punishing unused concepts:
        # summed = topic_vector_n.sum(axis = 0) #n_concept #want the mean of concept_probabilities to be bigger
        # return -torch.mean(summed)


    # def dotmm(self, a, b):
    #     #a: x, y, z
    #     #b: z, n
    #     return torch.einsum('xyz,zn->xyn', [a, b])
    
    def forward(self, f_input, method, targets, perturb = -1):
        f_input_n = F.normalize(f_input, dim = -1, p=2) #128, 100
        # print('f_input_n.shape: ', f_input_n.shape)
        topic_vector = self.topic_vector #100, 4
        # print('topic_vector.shape: ', topic_vector.shape)
        topic_vector_n = F.normalize(topic_vector, dim = 0, p=2) #100, 4
        # print('topic_vector_n.shape: ', topic_vector_n.shape)
        topic_prob = torch.mm(f_input, topic_vector_n) #bs, 4
        # print('topic_prob.shape: ', topic_prob.shape)
        topic_prob_n = torch.mm(f_input_n, topic_vector_n) #bs, 4
        # print('topic_prob_n.shape: ', topic_prob_n.shape)
        topic_prob_mask = torch.gt(topic_prob_n, self.thres) #128, 4
        # print('topic_prob_mask.shape: ', topic_prob_mask.shape)
        topic_prob_am = topic_prob * topic_prob_mask #128, 4
        # print('topic_prob_am.shape: ', topic_prob_am.shape)
        if perturb >=0:
            # print('topic_prob_am.shape: ', topic_prob_am.shape)
            topic_prob_am[:, perturb] = 0
        # +1e-3 to avoid division by zero
        topic_prob_sum = torch.sum(topic_prob_am, axis = -1, keepdim=True)+1e-3 #128, 1
        # print('topic_prob_sum.shape: ', topic_prob_sum.shape)
        topic_prob_nn = topic_prob_am / topic_prob_sum ##128, 4
        # print('topic_prob_nn.shape: ', topic_prob_nn.shape)
        rec_layer_1 = F.relu(torch.mm(topic_prob_nn, self.rec_vector_1)) #bs, 500
        # if self.args.extra_layers>0:
        #     for i in range(self.args.extra_layers):
        #         rec_layer_1 = F.relu(torch.mm(rec_layer_1, self.extra_recs[i]))
        # print('rec_layer_1.shape: ', rec_layer_1.shape)
        rec_layer_2 = torch.mm(rec_layer_1, self.rec_vector_2) #bs, 100
        # print('rec_layer_2.shape: ', rec_layer_2.shape) 
        # rec_layer_f2 = torch.flatten(rec_layer_2, 1)
        # print('rec_layer_f2.shape: ', rec_layer_f2.shape) #128, 39100
        if method == 'conceptshap':
            if self.args.divide_bert:
                rec_layer_2 = rec_layer_2.view(int(rec_layer_2.shape[0]/512), 512, 768)
                pred = self.classifier(inputs_embeds = rec_layer_2).logits
            else:
                pred = self.classifier(rec_layer_2) 
            if self.bert:
                pred = F.softmax(pred, dim = 1)
            # print('pred.shape: ', pred.shape) #128, 1
            concept_sim = self.concept_sim(topic_prob_n) # float
            concept_far = self.concept_far(topic_vector_n) #float
            # raise Exception('end')
            return pred, 0, concept_sim, concept_far, topic_prob_nn
        elif method == 'cc':
            if self.args.divide_bert:
                rec_layer_2 = rec_layer_2.view(int(rec_layer_2.shape[0]/512), 512, 768)
                pred = self.classifier(inputs_embeds = rec_layer_2).logits
            else:
                pred = self.classifier(rec_layer_2) 
            concept_sim = self.concept_sim(topic_prob_n) 
            concept_far = self.concept_far(topic_vector_n) 
            if self.bert:
                pred = F.softmax(pred, dim = 1)
            if self.args.masking != 'mean':
                if self.args.one_correlated_dimension == True:
                    original_last_dim = topic_prob_n[:, -1]
                    topic_prob_n = topic_prob_n[:, :-1]
                if self.args.masking=='max':
                    max_prob = torch.max(topic_prob_n, -1, keepdim=True).values #bs, 1
                    topic_prob_mask_far = torch.lt(topic_prob_n, max_prob) #if many of topic_prob_n are all zeros, will be all masked
                elif self.args.masking == 'random':
                    # print('topic_prob_n.shape: ', topic_prob_n.shape)
                    topic_prob_mask_far = (torch.cuda.FloatTensor(topic_prob_n.shape).uniform_() > self.args.random_masking_prob) #0.2 of zeros
                if self.args.one_correlated_dimension == True:
                    topic_prob_mask_new = torch.ones_like(topic_prob).to(self.device)
                    topic_prob_mask_new[:, :-1] = topic_prob_mask_far
                    topic_prob_mask_far = topic_prob_mask_new
                topic_prob_am_far = topic_prob * topic_prob_mask_far # topic_prob with zeroed out max dimension
                topic_prob_sum_far = torch.sum(topic_prob_am_far, axis=-1, keepdims=True)+1e-3 #bs, 1 #sum of the topic probabilities per instance
                topic_prob_nn_far = topic_prob_am_far/topic_prob_sum_far #normalized # normalize probabilities
                rec_layer_1_far = F.relu(torch.mm(topic_prob_nn_far, self.rec_vector_1))
                # if self.args.extra_layers>0:
                #     for i in range(self.args.extra_layers):
                #         rec_layer_1_far = F.relu(torch.mm(rec_layer_1_far, self.extra_recs[i]))
                rec_layer_2_far = torch.mm(rec_layer_1_far, self.rec_vector_2)
                if self.args.divide_bert:
                    rec_layer_2_far = rec_layer_2_far.view(int(rec_layer_2_far.shape[0]/512), 512, 768)
                    pred_perturbed = self.classifier(inputs_embeds = rec_layer_2_far).logits
                else:
                    pred_perturbed = self.classifier(rec_layer_2_far) 
                if self.bert:
                    pred_perturbed = F.softmax(pred_perturbed, dim = 1)
                flip_loss = self.flip_loss(pred, pred_perturbed, targets)
            else: #mean
                for c in range(self.n_concept):
                    topic_prob_mask_far = torch.ones_like(topic_prob_n).to(self.device) #batch_size, n_concept
                    topic_prob_mask_far[:, c] = 0
                    topic_prob_am_far = topic_prob * topic_prob_mask_far # topic_prob with zeroed out max dimension
                    topic_prob_sum_far = torch.sum(topic_prob_am_far, axis=-1, keepdims=True)+1e-3 #bs, 1 #sum of the topic probabilities per instance
                    topic_prob_nn_far = topic_prob_am_far/topic_prob_sum_far #normalized # normalize probabilities
                    rec_layer_1_far = F.relu(torch.mm(topic_prob_nn_far, self.rec_vector_1))
                    # if self.args.extra_layers>0:
                    #     for i in range(self.args.extra_layers):
                    #         rec_layer_1_far = F.relu(torch.mm(rec_layer_1_far, self.extra_recs[i]))
                    rec_layer_2_far = torch.mm(rec_layer_1_far, self.rec_vector_2)
                    if self.args.divide_bert:
                        rec_layer_2_far = rec_layer_2_far.view(int(rec_layer_2_far.shape[0]/512), 512, 768)
                        pred_perturbed = self.classifier(inputs_embeds = rec_layer_2_far).logits
                    else:
                        pred_perturbed = self.classifier(rec_layer_2_far) 
                    if self.bert:
                        pred_perturbed = F.softmax(pred_perturbed, dim = 1)
                    if c == 0:
                        flip_loss = self.flip_loss(pred, pred_perturbed, targets)
                    else:
                        flip_loss += self.flip_loss(pred, pred_perturbed, targets)
            # raise Exception('end')
            return pred, flip_loss, concept_sim, concept_far, topic_prob_nn

def concept_consistency(topic_prob_n, pred, y, batch_size):
    if isinstance(topic_prob_n, int):
        return 0
    # topic_prob_n: (bs, x, x, n_concept)
    # pred: (bs, n_classes)
    # find most similar regions
    loss = 0
    dims = len(topic_prob_n.shape) #4
    if dims > 2:
        for i in range(1, dims-1): # -1 to match indice number
            topic_prob_n = topic_prob_n.mean(1)
    if len(topic_prob_n.shape)!= 2:
        raise Exception('ended up with shape {}'.format(topic_prob_n.shape))
    for i in range(topic_prob_n.shape[-1]): #for each concept
        t = topic_prob_n[:, i]
        highest = torch.topk(t,k=batch_size//3,sorted=True).indices
        # find their corresponding preds
        p = torch.index_select(pred, 0, highest)
        # return distance squared
        pm = p.mean()
        # mean or max?
        loss += torch.square(p - pm).sum()
    return loss

##THIS ONE
class topic_model_toy(nn.Module):
    def __init__(self, criterion, classifier, f_train, n_concept, thres, device, args):
        super().__init__()
        # print(f_train.shape) #bs, 100, 391
        self.topic_vector = nn.Parameter(self.init_concept(f_train.shape[1], n_concept), requires_grad=True) #hidden_dim, 10
        # print('self.topic_vector.shape: ', self.topic_vector.shape)
        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concept, args.hidden_dim), requires_grad = True)
        # if args.extra_layers>0:
        #     self.extra_recs = [nn.Parameter(self.init_concept(args.hidden_dim, args.hidden_dim), requires_grad = True) for i in range(args.extra_layers)]
        self.rec_vector_2 = nn.Parameter(self.init_concept(args.hidden_dim, f_train.shape[1]), requires_grad = True)

        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.n_concept = n_concept
        self.thres = 0.3
        self.device = device
        if args.dataset!='toy':
            self.text = True
        else:
            self.text = False
        self.args = args
        self.criterion = criterion


    def init_concept(self, embedding_dim, n_concept):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concept) + r_1
        return concept
    
    def flip_loss(self, y_pred, pred_perturbed, targets):
        # maxmize the mean
        # minimize the negative
        # print('y_pred - pred_perturbed: ', y_pred - pred_perturbed)
        # print('torch.mean(torch.abs(y_pred - pred_perturbed)): ', torch.mean(torch.abs(y_pred - pred_perturbed)))
        # print('torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)): ', torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed))))
        # raise Exception('end')
        if self.args.loss=='flip':
            return -torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)))
        else:
            # print('here')
            loss = self.criterion(pred_perturbed.squeeze(), targets.squeeze().float()) 
            return - loss
        
    def concept_sim(self, topic_prob_n):
        # topic_prob_n : #32, 100, 4
        batch_size = topic_prob_n.shape[0]
        # print(topic_prob_n[0])
        res = torch.reshape(topic_prob_n,(-1,self.n_concept))
        # print('res.shape: ', res.shape) #3200, 4
        res = torch.transpose(res,0,1)
        # print('res.shape: ', res.shape) #4, 3200
        res = torch.topk(res,k=batch_size//4,sorted=True).values
        # print(res)
        # print('res.shape: ', res.shape)  #4, 32
        res = torch.mean(res)
        # print('res.shape: ', res.shape) 
        return - res

    def concept_far(self, topic_vector_n):
        # topic_vector_n: #hidden_dim, n_concept
        # after norm: n_concept, n_concept
        # base = torch.mm(torch.transpose(topic_vector_n, 0, 1), topic_vector_n) - torch.eye(self.n_concept).to(self.device)
        # fsm = torch.square(torch.mean(base))
        # ssm = torch.mean(torch.square(base))
        # return fsm + ssm
        return torch.mean(torch.mm(torch.transpose(topic_vector_n, 0, 1), topic_vector_n) - torch.eye(self.n_concept).to(self.device))

    def dotmm(self, a, b):
        #a: b, x, y, z
        #b: z, n
        if self.text:
            return torch.einsum('xyz,zn->xyn', [a, b])
        else:
            return torch.einsum('bxyz,zn->bxyn', [a, b])
    
    def forward(self, f_input, method, targets, perturb = -1):
        if self.text:
            n = 3
        else:
            n = 4
        torch.set_printoptions(profile="full")
        # input is 64, 4, 4, change to 4, 4, 64
        # print('Original f_input.shape: ', f_input.shape)
        # input is (bs, hidden_dim, maxlen), change to (bs, maxlen, hidden_dim)
        f_input = f_input.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
        # print('Swapped f_input.shape: ', f_input.shape)
        f_input_n = F.normalize(f_input, dim = n-1, p=2) #128, 100; 3 for toy, 2 for text
        # print('f_input_n.shape: ', f_input_n.shape)
        topic_vector = self.topic_vector #100, 4
        # print('topic_vector.shape: ', topic_vector.shape)
        # print('topic_vector: ', topic_vector)
        topic_vector_n = F.normalize(topic_vector, dim = 0, p=2) #100, 4
        # print('topic_vector_n: ', topic_vector_n)
        # print('f_input: ', f_input)
        topic_prob = self.dotmm(f_input, topic_vector_n) #bs, 4
        # print('topic_prob.shape: ', topic_prob.shape) #bs, maxlen, n_concept
        # print('topic_prob: ', topic_prob)
        # raise Exception('end')
        topic_prob_n = self.dotmm(f_input_n, topic_vector_n) #bs, 4
        # print('topic_prob_n.shape: ', topic_prob_n.shape)
        topic_prob_mask = torch.gt(topic_prob_n, self.thres) #128, 4
        # print('topic_prob_mask.shape: ', topic_prob_mask.shape)
        topic_prob_am = topic_prob * topic_prob_mask #128, 4
        if perturb >= 0:
            if not self.text:
                topic_prob_am[:, :, :, perturb] = 0
            else:
                # print('topic_prob_am.shape: ', topic_prob_am.shape) #size, sen_len, n_concept
                topic_prob_am[:, :, perturb] = 0
        # print('topic_prob_am.shape: ', topic_prob_am.shape)
        # +1e-3 to avoid division by zero
        topic_prob_sum = torch.sum(topic_prob_am, axis = n-1, keepdim=True)+1e-3 #128, 1; 3 for toy, 2 for text
        # print('topic_prob_sum.shape: ', topic_prob_sum.shape)
        topic_prob_nn = topic_prob_am / topic_prob_sum ##128, 4
        # print('topic_prob_nn.shape: ', topic_prob_nn.shape)
        rec_layer_1 = F.relu(self.dotmm(topic_prob_nn, self.rec_vector_1)) #bs, 500
        # if self.args.extra_layers>0:
        #     for i in range(self.args.extra_layers):
        #         rec_layer_1 = F.relu(torch.dotmm(rec_layer_1, self.extra_recs[i]))
        # print('rec_layer_1.shape: ', rec_layer_1.shape)
        rec_layer_2 = self.dotmm(rec_layer_1, self.rec_vector_2) #bs, 100
        # print('rec_layer_2.shape: ', rec_layer_2.shape) 
        # rec_layer_f2 = torch.flatten(rec_layer_2, 1)
        # print('rec_layer_f2.shape: ', rec_layer_f2.shape) #128, 39100
        if method == 'conceptshap':
            # print('Original rec_layer_2.shape: ', rec_layer_2.shape)
            rec_layer_2 = rec_layer_2.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
            # print('Swapped rec_layer_2.shape: ', rec_layer_2.shape)
            if self.text:
                rec_layer_2 = torch.mean(rec_layer_2, axis = -1) #bs, nc
            pred = self.classifier(rec_layer_2) 
            # print('pred.shape: ', pred.shape) #128, 1
            concept_sim = self.concept_sim(topic_prob_n) # float
            concept_far = self.concept_far(topic_vector_n) #float
            # raise Exception('end')
            return pred, 0, concept_sim, concept_far, topic_prob_nn
        elif method == 'cc':
            concept_sim = self.concept_sim(topic_prob_n) 
            concept_far = self.concept_far(topic_vector_n) 
            rec_layer_2 = rec_layer_2.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
            if self.text:
                rec_layer_2 = torch.mean(rec_layer_2, axis = -1) #bs, nc
            pred = self.classifier(rec_layer_2) 
            if self.args.masking != 'mean':
                if self.args.one_correlated_dimension == True:
                    original_last_dim = topic_prob_n[:, -1]
                    topic_prob_n = topic_prob_n[:, :-1]
                if self.args.masking=='max':
                    max_prob = torch.max(topic_prob_n, -1, keepdim=True).values #bs, 1
                    topic_prob_mask_far = torch.lt(topic_prob_n, max_prob) #if many of topic_prob_n are all zeros, will be all masked
                elif self.args.masking == 'random':
                    topic_prob_mask_far = (torch.cuda.FloatTensor(topic_prob_n.shape).uniform_() > self.args.random_masking_prob) #0.2 of zeros
                if self.args.one_correlated_dimension == True:
                    topic_prob_mask_new = torch.ones_like(topic_prob).to(self.device)
                    topic_prob_mask_new[:, :-1] = topic_prob_mask_far
                    topic_prob_mask_far = topic_prob_mask_new
                topic_prob_am_far = topic_prob * topic_prob_mask_far
                topic_prob_sum_far = torch.sum(topic_prob_am_far, axis=-1, keepdims=True)+1e-3 #bs, 1
                topic_prob_nn_far = topic_prob_am_far/topic_prob_sum_far
                rec_layer_1_far = F.relu(self.dotmm(topic_prob_nn_far, self.rec_vector_1))
                # if self.args.extra_layers>0:
                #     for i in range(self.args.extra_layers):
                #         rec_layer_1_far = F.relu(torch.dotmm(rec_layer_1_far, self.extra_recs[i]))
                rec_layer_2_far = self.dotmm(rec_layer_1_far, self.rec_vector_2)
                rec_layer_2_far = rec_layer_2_far.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
                if self.text:
                    rec_layer_2_far = torch.mean(rec_layer_2_far, axis = -1) #bs, nc
                pred_perturbed = self.classifier(rec_layer_2_far)
                flip_loss = self.flip_loss(pred, pred_perturbed, targets)
            else: #mean
                for c in range(self.n_concept):
                    topic_prob_mask_far = torch.ones_like(topic_prob_n).to(self.device) #batch_size, 4, 4, n_concept
                    if not self.text:
                        topic_prob_mask_far[:, :, :, c] = 0
                    else:
                        # print('topic_prob_am.shape: ', topic_prob_am.shape) #size, sen_len, n_concept
                        topic_prob_mask_far[:, :, c] = 0
                    topic_prob_am_far = topic_prob * topic_prob_mask_far
                    topic_prob_sum_far = torch.sum(topic_prob_am_far, axis=-1, keepdims=True)+1e-3 #bs, 1
                    topic_prob_nn_far = topic_prob_am_far/topic_prob_sum_far
                    rec_layer_1_far = F.relu(self.dotmm(topic_prob_nn_far, self.rec_vector_1))
                    # if self.args.extra_layers>0:
                    #     for i in range(self.args.extra_layers):
                    #         rec_layer_1_far = F.relu(torch.dotmm(rec_layer_1_far, self.extra_recs[i]))
                    rec_layer_2_far = self.dotmm(rec_layer_1_far, self.rec_vector_2)
                    rec_layer_2_far = rec_layer_2_far.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
                    if self.text:
                        rec_layer_2_far = torch.mean(rec_layer_2_far, axis = -1) #bs, nc
                    pred_perturbed = self.classifier(rec_layer_2_far)
                    if c == 0:
                        flip_loss = self.flip_loss(pred, pred_perturbed)
                    else:
                        flip_loss += self.flip_loss(pred, pred_perturbed)
            return pred, flip_loss, concept_sim, concept_far, topic_prob_nn
            