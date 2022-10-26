using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class MnistInference
{
    private readonly InferenceSession session;

    public MnistInference(string modelPath)
    {
        session = new InferenceSession(modelPath);
    }

    public int Inference(float[] input_floats)
    {
        //���_����
        var scores = InferenceOnnx(input_floats);

        //�ő��Index�����߂�DIndex�����_��������
        var maxScore = float.MinValue;
        int maxIndex = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            float score = scores[i];
            if (maxScore < score)
            {
                maxScore = score;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private float[] InferenceOnnx(float[] input)
    {
        var inputName = session.InputMetadata.First().Key;
        var inputDim = session.InputMetadata.First().Value.Dimensions;
        var inputTensor = new DenseTensor<float>(new System.Memory<float>(input), inputDim);

        // OnnxRuntime�ł̓��͌`���ł���NamedOnnxValue���쐬����
        var inputOnnxValues = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor (inputName, inputTensor)
        };

        // ���_�����s
        var results = session.Run(inputOnnxValues);
        var scores = results.First().AsTensor<float>().ToArray();

        return scores;
    }
}
