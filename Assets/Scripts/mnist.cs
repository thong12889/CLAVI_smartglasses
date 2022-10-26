using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;

public class mnist : MonoBehaviour
{
    [SerializeField] private Text text;

    // Start is called before the first frame update
    void Start()
    {
        var path = Path.Combine(Application.streamingAssetsPath, "mnist-8.onnx"); 
        var model = File.ReadAllBytes(path);

        var session = new InferenceSession(model);

        // 28x28 grayscale 
        var input = new float[784];

        var inputName = session.InputMetadata.First().Key;
        var inputDim = session.InputMetadata.First().Value.Dimensions;
        var inputTensor = new DenseTensor<float>(new Memory<float>(input), inputDim);

        var inputOnnxValues = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

        var results = session.Run(inputOnnxValues);
        var scores = results.First().AsTensor<float>().ToArray();

        text.text = string.Join(",", scores);
        foreach (var score in scores) Debug.Log(score);
    }
}
