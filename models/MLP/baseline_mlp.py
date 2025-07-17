from models.MLP.soft_hebbian_layer import SoftHebbLayer, SoftNeuralNet
from models.utils.hyperparams import LearningRule


def MLPBaseline(
    K, epsilon, focus, hsize, lamb, w_lr, b_lr, l_lr, nclasses, device, weight_growth
):
    mymodel = SoftNeuralNet(device, hsize)
    heb_layer = SoftHebbLayer(
        K=K,
        epsilon=epsilon,
        focus=focus,
        inputdim=784,
        outputdim=hsize,
        w_lr=w_lr,
        b_lr=b_lr,
        l_lr=l_lr,
        device=device,
        initial_lambda=lamb,
        weight_growth=weight_growth,
    )

    heb_layer2 = SoftHebbLayer(
        K=K,
        epsilon=epsilon,
        focus=focus,
        inputdim=hsize,
        outputdim=nclasses,
        w_lr=w_lr,
        b_lr=b_lr,
        l_lr=l_lr,
        initial_lambda=lamb,
        weight_growth=weight_growth,
        learningrule=LearningRule.SoftHebbOutputContrastive,
        is_output_layer=True,
    )
    mymodel.add_layer("SoftHebbian1", heb_layer)
    mymodel.add_layer("SoftHebbian2", heb_layer2)

    return mymodel
