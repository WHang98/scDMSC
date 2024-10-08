import torch
n=30672        #样本个数
def loss_fn(self,x,x1,x2,x1_rec,x2_rec,z, zc, weight_ae,weight_trace, weight_selfExp,weight_C):

        loss1_ae = weight_ae*torch.sum(torch.square(torch.subtract(x1_rec, x1)))
        loss2_ae = weight_ae*torch.sum(torch.square(torch.subtract(x2_rec, x2)))
        loss_ae=loss1_ae +loss2_ae  
        x1_inputfla = torch.reshape(x1, [n, -1])
        x1_inputfla = x1_inputfla.to(torch.float32)
        x1_recfla = torch.reshape(x1_rec, [n, -1])
        x1_recfla = x1_recfla.to(torch.float32)
        x2_inputfla = torch.reshape(x2, [n, -1])
        x2_inputfla = x2_inputfla.to(torch.float32)
        x2_recfla = torch.reshape(x2_rec, [n, -1])
        x2_recfla = x2_recfla.to(torch.float32)
        normL = True
        absC = torch.abs(self.self_expression.Coefficient)
        C = (absC + absC.T) * 0.5
        C = C + torch.eye(self.self_expression.Coefficient.shape[0])

        if normL == True:
            D = torch.diag(1.0 / torch.sum(C,axis=1))
            I = torch.eye(D.shape[0])
            L = I - torch.matmul(D,C)
            D = I
        else:
            D = torch.diag(torch.sum(C, axis=1))
            L = D - C
        XLX_r1 = torch.matmul(torch.matmul((x1_inputfla.T),L),x1_recfla)
        XLX_r2 = torch.matmul(torch.matmul((x2_inputfla.T),L),x2_recfla)
        X1sub = x1_inputfla - x1_recfla
        X2sub = x2_inputfla - x2_recfla
        tracelossx1 =torch.sum(torch.square(X1sub)) +  2.0 * torch.trace(XLX_r1)#/self.batch_size
        tracelossx2 =torch.sum(torch.square(X2sub)) +  2.0 * torch.trace(XLX_r2)#/self.batch_size
        tracelossx =tracelossx1+ tracelossx2
        loss_selfExp = torch.sum(torch.square(torch.subtract(zc, z)))
        norm = torch.norm(self.self_expression.Coefficient,keepdim=True)
        penalty = torch.matmul(norm, norm)
        loss_s=torch.sum(torch.abs(self.self_expression.Coefficient))+ torch.trace(torch.abs(self.self_expression.Coefficient))
        loss_sc = weight_ae*loss_ae + weight_trace * tracelossx+ weight_selfExp * loss_selfExp#+penalty+loss_s#+weight_C*Contrast
        loss_sc /= x1.size(0)  # just control the range, does not affect the optimization.
        return loss_sc,loss_ae, tracelossx,loss_selfExp,penalty,loss_s