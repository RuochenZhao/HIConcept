# Concept Models
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

##THIS ONE
class topic_model_main(nn.Module):
    def __init__(self, criterion, classifier, f_train, n_concept, thres, device, args):
        super().__init__()
        if args.init_with_pca:
            self.topic_vector = nn.Parameter(self.init_with_pca(f_train, n_concept), requires_grad=True)
        else:
            self.topic_vector = nn.Parameter(self.init_concept(f_train.shape[-1], n_concept), requires_grad=True) #hidden_dim, 10
        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concept, args.hidden_dim), requires_grad = True)
        self.rec_vector_2 = nn.Parameter(self.init_concept(args.hidden_dim, f_train.shape[-1]), requires_grad = True)

        self.classifier = classifier
        self.args = args
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.n_concept = n_concept
        if self.args.overall_method == 'conceptshap':
            self.thres = 0.3
        else:
            self.thres = 0.1
        self.device = device
        self.bert = (args.model_name=='bert')
        self.criterion = criterion
        self.ae_criterion = nn.MSELoss()

    def init_with_pca(self, f_train, n_concept):
        pca = PCA(n_components = n_concept)
        pca.fit(f_train.numpy().astype(float))
        weight_pca = torch.from_numpy(pca.components_.T) #hidden_dim, n_concept
        return weight_pca.float()

    def init_concept(self, embedding_dim, n_concept):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concept) + r_1
        return concept
    
    def ae_loss(self, xpred, xin):
        return self.ae_criterion(xpred, xin)

    def flip_loss(self, y_pred, pred_perturbed, targets):
        if self.args.loss=='flip':
            if self.bert or self.args.model_name == 't5':
                y_pred = F.softmax(y_pred, dim = 1)[:, 1]
                pred_perturbed = F.softmax(pred_perturbed, dim = 1)[:, 1]
                return -torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)))
            return -torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)))
        else:
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

    def concept_far(self, topic_vector_n):
        # topic_vector_n: #hidden_dim, n_concept
        # after norm: n_concept, n_concept
        return torch.mean(torch.mm(torch.transpose(topic_vector_n, 0, 1), topic_vector_n) - torch.eye(self.n_concept).to(self.device))
        
    def forward(self, f_input, causal, targets, perturb = -1):
        f_input_n = F.normalize(f_input, dim = -1, p=2) #128, 100
        topic_vector = self.topic_vector #100, 4
        topic_vector_n = F.normalize(topic_vector, dim = 0, p=2) #100, 4
        topic_prob = torch.mm(f_input, topic_vector_n) #bs, 4
        topic_prob_n = torch.mm(f_input_n, topic_vector_n) #bs, 4
        if self.args.overall_method!='conceptshap':
            topic_prob_n = F.softmax(topic_prob_n, dim = -1)
        topic_prob_mask = torch.gt(topic_prob_n, self.thres) #128, 4
        topic_prob_am = topic_prob * topic_prob_mask #128, 4
        if perturb >=0:
            topic_prob_am[:, perturb] = 0
        # +1e-3 to avoid division by zero
        topic_prob_sum = torch.sum(topic_prob_am, axis = -1, keepdim=True)+1e-3 #128, 1
        topic_prob_nn = topic_prob_am / topic_prob_sum ##128, 4
        rec_layer_1 = F.relu(torch.mm(topic_prob_nn, self.rec_vector_1)) #bs, 500
        rec_layer_2 = torch.mm(rec_layer_1, self.rec_vector_2) #bs, 100
        ae_loss = self.ae_loss(f_input_n, rec_layer_2)
        if self.args.divide_bert:
            rec_layer_2 = rec_layer_2.view(int(rec_layer_2.shape[0]/512), 512, 768)
            pred = self.classifier(inputs_embeds = rec_layer_2).logits
        else:
            pred = self.classifier(rec_layer_2)
        concept_sim = self.concept_sim(topic_prob_n) 
        concept_far = self.concept_far(topic_vector_n) 
        if causal != True:
            return pred, 0, concept_sim, concept_far, topic_prob_nn, ae_loss
        else:
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
                rec_layer_2_far = torch.mm(rec_layer_1_far, self.rec_vector_2)
                if self.args.divide_bert:
                    rec_layer_2_far = rec_layer_2_far.view(int(rec_layer_2_far.shape[0]/512), 512, 768)
                    pred_perturbed = self.classifier(inputs_embeds = rec_layer_2_far).logits
                else:
                    pred_perturbed = self.classifier(rec_layer_2_far) 
                if self.bert or self.args.model_name == 't5':
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
                    rec_layer_2_far = torch.mm(rec_layer_1_far, self.rec_vector_2)
                    if self.args.divide_bert:
                        rec_layer_2_far = rec_layer_2_far.view(int(rec_layer_2_far.shape[0]/512), 512, 768)
                        pred_perturbed = self.classifier(inputs_embeds = rec_layer_2_far).logits
                    else:
                        pred_perturbed = self.classifier(rec_layer_2_far) 
                    if self.bert or self.args.model_name == 't5':
                        pred_perturbed = F.softmax(pred_perturbed, dim = 1)
                    if c == 0:
                        flip_loss = self.flip_loss(pred, pred_perturbed, targets)
                    else:
                        flip_loss += self.flip_loss(pred, pred_perturbed, targets)
            return pred, flip_loss, concept_sim, concept_far, topic_prob_nn, ae_loss

##THIS ONE
class topic_model_toy(nn.Module):
    def __init__(self, criterion, classifier, f_train, n_concept, thres, device, args):
        super().__init__()
        self.args = args
        args.logger.info(f'f_train.shape: {f_train.shape}') #bs, 100, 391
        if args.init_with_pca:
            self.topic_vector = nn.Parameter(self.init_with_pca(f_train, n_concept), requires_grad=True)
        else:
            if args.model_name == 'transformer':
                raise Exception('should not be here!')
            else:
                self.topic_vector = nn.Parameter(self.init_concept(f_train.shape[1], n_concept), requires_grad=True) #hidden_dim, 10
        print('self.topic_vector.shape: ', self.topic_vector.shape)
        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concept, args.hidden_dim), requires_grad = True)
        self.rec_vector_2 = nn.Parameter(self.init_concept(args.hidden_dim, f_train.shape[1]), requires_grad = True)

        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.n_concept = n_concept
        if self.args.overall_method == 'conceptshap':
            self.thres = 0.3
        else:
            self.thres = 0.1
        self.device = device
        if args.dataset!='toy':
            self.text = True
        else:
            self.text = False
        self.criterion = criterion
        self.ae_criterion = nn.MSELoss()


    def init_with_pca(self, f_train, n_concept):
        pca = PCA(n_components = n_concept)
        if self.args.dataset == 'toy':
            n = 4
        else:
            n = 3
        self.args.logger.info(f'Original f_train.shape: {f_train.shape}')
        f_train = f_train.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
        self.args.logger.info(f'Swapped f_train.shape: {f_train.shape}')
        if f_train.dim()>2:
            f_train = f_train.flatten(start_dim = 0, end_dim = -2)
        pca.fit(f_train.numpy().astype(float))
        weight_pca = torch.from_numpy(pca.components_.T) #hidden_dim, n_concept
        return weight_pca.float()

    def init_concept(self, embedding_dim, n_concept):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concept) + r_1
        return concept
    
    def ae_loss(self, xpred, xin):
        ae_l = self.ae_criterion(xpred, xin)
        return ae_l

    def flip_loss(self, y_pred, pred_perturbed, targets):
        if self.args.loss=='flip':
            return -torch.mean(torch.mean(torch.abs(y_pred - pred_perturbed)))
        else:
            loss = self.criterion(pred_perturbed.squeeze(), targets.squeeze().float()) 
            return - loss
        
    def concept_sim(self, topic_prob_n):
        # topic_prob_n : #32, 100, 4
        batch_size = topic_prob_n.shape[0]
        res = torch.reshape(topic_prob_n,(-1,self.n_concept))
        res = torch.transpose(res,0,1)
        res = torch.topk(res,k=batch_size//4,sorted=True).values
        res = torch.mean(res)
        return - res

    def concept_far(self, topic_vector_n):
        # topic_vector_n: #hidden_dim, n_concept
        # after norm: n_concept, n_concept
        return torch.mean(torch.mm(torch.transpose(topic_vector_n, 0, 1), topic_vector_n) - torch.eye(self.n_concept).to(self.device))

    def dotmm(self, a, b):
        #a: b, x, y, z
        #b: z, n
        if self.text:
            return torch.einsum('xyz,zn->xyn', [a, b])
        else:
            return torch.einsum('bxyz,zn->bxyn', [a, b])
    
    def forward(self, f_input, causal, targets, perturb = -1):
        if self.text:
            n = 3
        else:
            n = 4
        torch.set_printoptions(profile="full")
        # input is 64, 4, 4, change to 4, 4, 64
        if self.args.model_name != 'transformer':
            f_input = f_input.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
        f_input_n = F.normalize(f_input, dim = n-1, p=2) #128, 100; 3 for toy, 2 for text
        topic_vector = self.topic_vector #100, 4
        topic_vector_n = F.normalize(topic_vector, dim = 0, p=2) #100, 4
        topic_prob = self.dotmm(f_input, topic_vector_n) #bs, 4
        topic_prob_n = self.dotmm(f_input_n, topic_vector_n) #bs, 4
        if self.args.overall_method!='conceptshap':
            topic_prob_n = F.softmax(topic_prob_n, dim = -1)
        topic_prob_mask = torch.gt(topic_prob_n, self.thres) #128, 4
        topic_prob_am = topic_prob * topic_prob_mask #128, 4
        if perturb >= 0:
            if not self.text:
                topic_prob_am[:, :, :, perturb] = 0
            else:
                topic_prob_am[:, :, perturb] = 0
        # +1e-3 to avoid division by zero
        topic_prob_sum = torch.sum(topic_prob_am, axis = n-1, keepdim=True)+1e-3 #128, 1; 3 for toy, 2 for text
        topic_prob_nn = topic_prob_am / (topic_prob_sum+1e-8) ##128, 4
        rec_layer_1 = F.relu(self.dotmm(topic_prob_nn, self.rec_vector_1)) #bs, 500
        rec_layer_2 = self.dotmm(rec_layer_1, self.rec_vector_2) #bs, 100
        ae_loss = self.ae_loss(f_input_n, rec_layer_2)
        if self.args.model_name != 'transformer':
            rec_layer_2 = rec_layer_2.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
        if self.text and self.args.model_name =='cnn':
            rec_layer_2 = torch.mean(rec_layer_2, axis = -1) #bs, nc
        try:
            pred = self.classifier(rec_layer_2) 
        except:
            print(f'rec_layer_2: {rec_layer_2}')
            print('topic_prob_n: ', topic_prob_n)
            print('topic_prob_n: ', topic_prob_n)
            raise Exception('end')
        concept_sim = self.concept_sim(topic_prob_n) 
        concept_far = self.concept_far(topic_vector_n) 
        
        if causal != True:
            return pred, 0, concept_sim, concept_far, topic_prob_nn, ae_loss
        else:
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
                rec_layer_2_far = self.dotmm(rec_layer_1_far, self.rec_vector_2)
                if self.args.model_name != 'transformer':
                    rec_layer_2_far = rec_layer_2_far.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
                if self.text and self.args.model_name =='cnn':
                    rec_layer_2_far = torch.mean(rec_layer_2_far, axis = -1) #bs, nc
                pred_perturbed = self.classifier(rec_layer_2_far)

                flip_loss = self.flip_loss(pred, pred_perturbed, targets)
            else: #mean
                for c in range(self.n_concept):
                    topic_prob_mask_far = torch.ones_like(topic_prob_n).to(self.device) #batch_size, 4, 4, n_concept
                    if not self.text:
                        topic_prob_mask_far[:, :, :, c] = 0
                    else:
                        topic_prob_mask_far[:, :, c] = 0
                    topic_prob_am_far = topic_prob * topic_prob_mask_far
                    topic_prob_sum_far = torch.sum(topic_prob_am_far, axis=-1, keepdims=True)+1e-3 #bs, 1
                    topic_prob_nn_far = topic_prob_am_far/topic_prob_sum_far
                    rec_layer_1_far = F.relu(self.dotmm(topic_prob_nn_far, self.rec_vector_1))
                    rec_layer_2_far = self.dotmm(rec_layer_1_far, self.rec_vector_2)
                    if self.args.model_name != 'transformer':
                        rec_layer_2_far = rec_layer_2_far.swapaxes(1, n-1) #1, 3 for toy; 1, 2 for text
                    if self.text and self.args.model_name =='cnn':
                        rec_layer_2_far = torch.mean(rec_layer_2_far, axis = -1) #bs, nc
                    pred_perturbed = self.classifier(rec_layer_2_far)
                    if c == 0:
                        flip_loss = self.flip_loss(pred, pred_perturbed)
                    else:
                        flip_loss += self.flip_loss(pred, pred_perturbed)
            return pred, flip_loss, concept_sim, concept_far, topic_prob_nn, ae_loss
            