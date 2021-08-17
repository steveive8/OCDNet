import torch
import numpy as np

#========================================================================
#https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html=
# export 함수가 모델을 실행하기 때문에, 우리가 직접 텐서를 입력값으로 넘겨주어야 합니다. 
# 이 텐서의 값은 알맞은 자료형과 모양이라면 랜덤하게 결정되어도 무방
# 특정 차원을 동적인 차원으로 지정하지 않는 이상, 
# ONNX로 변환된 그래프의 경우 입력값의 사이즈는 모든 차원에 대해 고정됩니다. 
# 예시에서는 모델이 항상 배치 사이즈 1을 사용하도록 변환하였지만, 
# 첫번째 차원을 torch.onnx.export() 의 dynamic_axes 인자에 동적인 차원으로 지정

# https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html
# 모델을 변환하기 전에 모델을 추론 모드로 바꾸기 위해서
#  torch_model.eval() 또는 torch_model.train(False) 를 호출하는 것이 중요합니다.
#   이는 dropout이나 batchnorm과 같은 연산들이
#   추론과 학습 모드에서 다르게 작동하기 때문에 필요합니다.
# ex) 모델을 추론 모드로 전환합니다
# torch_model.eval()
# __________________________________________________________________________
def SaveONNX(nModel, nFileOnnx, nVer, nChannel, nImageSize):

    print("\n \n **** SaveONNX =",nFileOnnx )

    # Neural network input data type
    nBatchSize = 1
    #nInput_Shape = torch.randn(1, 1, 28, 28, device='cuda')
    nInput_Shape = torch.randn(nBatchSize, nChannel, nImageSize, nImageSize, device='cuda')

    # ONNX 런타임에서 변환된 모델을 사용했을 때 같은 결과를 얻는지 확인하기 위해서 torch_out 를 계산합니다.
    torch_out = nModel(nInput_Shape)

    # It's optional to label the input and output layers
    #input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    nInput_names = [ "Cnn_Input" ]
    nOutput_names = [ "Cnn_Output" ]

    # https://github.com/onnx/onnx/blob/master/docs/Versioning.md
    # Opset version ai.onnx 1,5,6,7,8,9,10,11,12,13
    if nVer > 3 :    
        torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=False, opset_version=nVer, input_names=nInput_names, output_names=nOutput_names)
    else :
        torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=True, input_names=nInput_names, output_names=nOutput_names)
    #torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=True)

    import onnx

    print("\n \n **** ONNX  Check ")
    onnx_model = onnx.load(nFileOnnx)
    # ONNX 그래프의 유효성은 모델의 버전, 그래프 구조, 노드들, 그리고 입력값과 출력값들을 모두 체크하여 결정
    onnx.checker.check_model(onnx_model)
    print("Success - model_onnx was saved !")

    print("그래프 포맷으로 출력 !")
    onnx.helper.printable_graph(onnx_model.graph)


    # ONNX 런타임의 Python API를 통해 결과값을 계산해보도록 하겠습니다. 
    # 이 부분은 보통 별도의 프로세스 또는 별도의 머신에서 실행되지만,
    # 모델이 ONNX 런타임과 PyTorch에서 동일한 결과를 출력하는지를 확인하기 위해 
    # 동일한 프로세스에서 계속 실행하도록 하겠습니다.
    # 
    # 모델을 ONNX 런타임에서 실행하기 위해서는 
    # 미리 설정된 인자들(본 예제에서는 기본값을 사용합니다)로 모델을 위한 추론 세션을 생성 해야 합니다.
    # 세션이 생성되면, 모델의 run() API를 사용하여 모델을 실행합니다. 
    # 이 API의 리턴값은 ONNX 런타임에서 연산된 모델의 결과값들을 포함하고 있는 리스트입니다.-

    import onnxruntime

    print("ONNX 런타임과 PyTorch에서 연산된 결과값 비교")
    ort_session = onnxruntime.InferenceSession(nFileOnnx)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(nInput_Shape)}
    ort_outs = ort_session.run(None, ort_inputs)

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    # PyTorch와 ONNX 런타임에서 연산된 결과값이 서로 일치하는지
    # 오차범위 (rtol=1e-03, atol=1e-05) 이내에서 확인해야 합니다.
    # 만약 결과가 일치하지 않는다면 ONNX 변환기에 문제가 있는 것이니 저희에게 알려주시기 바랍니다.



    print("====== onnxsim import simplify !")
    nFileOnnxSimple = nFileOnnx.replace(".onnx","_sim.onnx")

    import os 
    os.system("python -m onnxsim {0} {1}".format(nFileOnnx, nFileOnnxSimple))

    """
    # https://github.com/daquexian/onnx-simplifier
    from onnxsim import simplify

    # load your predefined ONNX model
    model = onnx.load(nFileOnnx)


    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    # use model_simp as a standard ONNX model object
    nFileOnnxSimple = text.replace(".onnx","_sim.onnx")
    #nFileOnnxSimple = nFilepath + "_sim.onnx"
    torch.onnx.export(model_simp, nInput_Shape, nFileOnnxSimple)
    """

    #python -m onnxsim ./Model/Mnist.onnx ./Model/Mnist_s.onnx
    #python -m onnxsim ./Model/Cifar10_t_n.onnx ./Model/Cifar10_t_s.onnx
    
# __________________________________________________________________________
def SaveONNX_Dynamic(nModel, nFileOnnx, nVer, nChannel, nImageSize):
    print("\n \n **** SaveONNX =",nFileOnnx )

    # Neural network input data type
    nBatchSize = 1
    #nInput_Shape = torch.randn(1, 1, 28, 28, device='cuda')
    nInput_Shape = torch.randn(nBatchSize, nChannel, nImageSize, nImageSize, device='cuda')

    # ONNX 런타임에서 변환된 모델을 사용했을 때 같은 결과를 얻는지 확인하기 위해서 torch_out 를 계산합니다.
    torch_out = nModel(nInput_Shape)

    # It's optional to label the input and output layers
    #input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    nInput_names = [ "Cnn_Input" ]
    nOutput_names = [ "Cnn_Output" ]

    dynamic_axes={"Cnn_Input": {0: "batch"}, "Cnn_Output": {0: "batch"}}

    # https://github.com/onnx/onnx/blob/master/docs/Versioning.md
    # Opset version ai.onnx 1,5,6,7,8,9,10,11,12,13

    if nVer > 3 :    
        torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=False, opset_version=nVer, do_constant_folding=True, 
                          input_names=nInput_names, output_names=nOutput_names, dynamic_axes=dynamic_axes)
        #torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=False, opset_version=nVer, input_names=nInput_names, output_names=nOutput_names, dynamic_axes=dynamic_axes)
        #torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=False, opset_version=nVer, input_names=nInput_names, output_names=nOutput_names)
    else :
        torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=True, do_constant_folding=True, input_names=nInput_names, output_names=nOutput_names, dynamic_axes=dynamic_axes)
        #torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=True, input_names=nInput_names, output_names=nOutput_names, dynamic_axes=dynamic_axes)
        #torch.onnx.export(nModel, nInput_Shape, nFileOnnx, verbose=True, input_names=nInput_names, output_names=nOutput_names)

    """
    #https://www.jianshu.com/p/36ff0e224112
    x = torch.onnx.export(nModel,  # 변환 할 네트워크 모델 
                    torch.randn(1, 1, nImageSize, nImageSize, device='cuda'), # 추론 계산 그래프에서 각 노드의 입력 크기와 크기를 결정하는 데 사용되는 가상 입력
                    nFileOnnx,  # 출력 파일의 이름
                    verbose=False,      # 계산 그래프를 문자열 형식으로 표시할지 여부
                    input_names=["input"],# + ["params_%d"%i for i in range(120)],  # 입력 노드의 이름
                    output_names=["output"], # 출력 노드의 이름
                    opset_version=10,   # 현재 최대 10 개까지 지원합니다.(TensorRT 7 ONNX에서)
                    do_constant_folding=True, # 상수 압축 여부
                    dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} #동적 차원을 설정합니다. 여기서는 입력 노드의 0 번째 차원이 변수 인 batch_size라는 것을 나타냅니다.
                    )
    """

    import onnx

    print("\n \n **** ONNX  Check ")
    onnx_model = onnx.load(nFileOnnx)
    # ONNX 그래프의 유효성은 모델의 버전, 그래프 구조, 노드들, 그리고 입력값과 출력값들을 모두 체크하여 결정
    onnx.checker.check_model(onnx_model)
    print("Success - model_onnx was saved !")

    print("그래프 포맷으로 출력 !")
    onnx.helper.printable_graph(onnx_model.graph)


    # ONNX 런타임의 Python API를 통해 결과값을 계산
    # 이 부분은 보통 별도의 프로세스 또는 별도의 머신에서 실행되지만,
    # 모델이 ONNX 런타임과 PyTorch에서 동일한 결과를 출력하는지를 확인하기 위해 
    # 동일한 프로세스에서 계속 실행하도록 하겠습니다.
    # 
    # 모델을 ONNX 런타임에서 실행하기 위해서는 
    # 미리 설정된 인자들(본 예제에서는 기본값을 사용합니다)로 모델을 위한 추론 세션을 생성 해야 합니다.
    # 세션이 생성되면, 모델의 run() API를 사용하여 모델을 실행합니다. 
    # 이 API의 리턴값은 ONNX 런타임에서 연산된 모델의 결과값들을 포함하고 있는 리스트입니다.-

    import onnxruntime

    print("ONNX 런타임과 PyTorch에서 연산된 결과값 비교")
    ort_session = onnxruntime.InferenceSession(nFileOnnx)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(nInput_Shape)}
    ort_outs = ort_session.run(None, ort_inputs)

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    # PyTorch와 ONNX 런타임에서 연산된 결과값이 서로 일치하는지
    # 오차범위 (rtol=1e-03, atol=1e-05) 이내에서 확인해야 합니다.
    # 만약 결과가 일치하지 않는다면 ONNX 변환기에 문제가 있는 것이니 저희에게 알려주시기 바랍니다.


    print("\n\n ONNX 런타임에서 계산된 결과값")
    print("\n\n ====== Dynamic ONNX < simplify > Dynamic 오케이 ...!")





#========================================================================
#
# __________________________________________________________________________
def SaveTracedModel(nModel, nFilepath, nChannel, nImageSize ):
    # Set upgrading the gradients to False
    #for param in model.parameters():
	#    param.requires_grad = False

    nBatchSize = 1
    nRandomInput_Shape = torch.randn(nBatchSize, nChannel, nImageSize, nImageSize, device='cuda')
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    print("\n \n **** SaveTracedModel =",nFilepath )
    traced_script_module = torch.jit.trace(nModel, nRandomInput_Shape)
    traced_script_module.save(nFilepath)
    print("Success - model_trace was saved !")
# __________________________________________________________________________
def LoadTracedModel(nFilepath):
    print("\n \n **** LoadTracedModel =",nFilepath )
    traced_script_module = torch.jit.load(nFilepath)
    print("Success - model_trace was load !")
    return traced_script_module

# __________________________________________________________________________
def SaveModelPth(nModel, nFilepath):
    print("\n \n **** SaveModelPth =",nFilepath )

    torch.save(nModel.state_dict(), nFilepath)
    print("Success - model_trace was saved !")

# __________________________________________________________________________
def LoadModelPth(nModel, nFilepath):
    print("\n \n **** LoadModelPth =",nFilepath )

    nModel.load_state_dict(torch.load(nFilepath))
    print("Success - model_trace was load !")
    return nModel

# __________________________________________________________________________
def SaveTxtPth(nFilepath, strTxt):
    strTxt += "\n"
    with open(nFilepath, 'a') as f:
        f.write(strTxt)


# ========================================================================
def SaveModel(net, sCkptPath, sModelName, nChannel, nCnnSize):
    nOnnxVer = 10

    #__________________________________________________________________________
    # Model Save
    print("\n \n **** SaveTracedModel jit.scrip ")
    # 일반적으로 모델의 forward() 메소드에 넘겨주는 입력값
    nFilepath = sCkptPath + "/" + sModelName + ".pt"
    SaveTracedModel(net, nFilepath, nChannel, nCnnSize)

    print("\n \n **** 모델을 onnx -----------")
    #sCkptFile = sCkptPath + "/ckpt.pth"

    #checkpoint = torch.load(sCkptFile)
    #net.load_state_dict(checkpoint['State_dict'])


    #print("\n \n **** 모델을 Dynamic ONNX ")
    #nOnnxFile = sCkptPath + "/" + sModelName + "_d.onnx"
    #SaveONNX_Dynamic(net, nOnnxFile, nOnnxVer, nChannel, nCnnSize)

    #nOnnxFile = sCkptPath + "/" + sModelName + ".onnx"
    #SaveONNX(net, nOnnxFile, nOnnxVer, nChannel, nCnnSize)
    #__________________________________________________________________________

import ocd_1

net = ocd_1.OCD10(num_classes = 1000, init_weights = True).to('cuda')

test = torch.rand(1, 3, 180, 180).to('cuda')

out = net(test)

SaveModel(net, 'models', 'ocd_1', 3, 180)
