using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEngine.UI;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

public class obj_nocv : MonoBehaviour
{
    public RawImage rawimage;
    [SerializeField] TextAsset labelMap = null;

    private InferenceSession session;
    private int imgW = 640;
    private int imgH = 640;
    private string[] labelList;
    private WebCamTexture webcamTexture;

    // Start is called before the first frame update
    void Start()
    {
        labelList = labelMap.text.Split('\n');

        webcamTexture = new WebCamTexture(imgH, imgW);
        rawimage.texture = webcamTexture;
        webcamTexture.Play();

        BetterStreamingAssets.Initialize();
        byte[] model = BetterStreamingAssets.ReadAllBytes("board.onnx");
        var option = new SessionOptions();
        //option.AppendExecutionProvider_Nnapi(NnapiFlags.NNAPI_FLAG_CPU_DISABLED);
        option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
        session = new InferenceSession(model, option);
    }

    // Update is called once per frame
    void Update()
    {
        if (webcamTexture == null)
        {
            return;
        }
        else
        {
            var floatSize = imgW * imgH * 3;
            var color = webcamTexture.GetPixels32();
            float[] channelData = new float[floatSize];
            var expectedOutputLength = floatSize;
            var expectedChannelLength = expectedOutputLength / 3;
            var greenOffset = expectedChannelLength;
            var blueOffset = expectedChannelLength * 2;
            var aOffset = expectedChannelLength * 3;
            for (int i = 0; i < color.Length; i++)
            {
                /*var r = Convert.ToSingle(color[i].r);
                var g = Convert.ToSingle(color[i].g);
                var b = Convert.ToSingle(color[i].b);
                var a = Convert.ToSingle(color[i].a);*/
                var r = color[i].r;
                var g = color[i].g;
                var b = color[i].b;
                var a = color[i].a;
                channelData[i] = r;
                channelData[i + 1] = g;
                channelData[i + 2] = b;
                //channelData[i + aOffset] = a;
            }
            var byteArray = new byte[channelData.Length * 3];
            Buffer.BlockCopy(channelData, 0, byteArray, 0, byteArray.Length);
            //var byteArray = channelData.Select(f => Convert.ToByte(f)).ToArray();
            var texture2d = new Texture2D(imgW, imgH, TextureFormat.RGB24, false);
            texture2d.LoadRawTextureData(byteArray);
            texture2d.Apply();
            rawimage.texture = texture2d;
            //Process(channelData);
        }
    }
    private void Process(float[] image)
    {
        //image.Resize(imgW, imgH);
        var input = new DenseTensor<float>(image, new[] { 1, 3, imgH, imgW });

        // Setup inputs and outputs
        var inputMeta = session.InputMetadata;
        var inputName = inputMeta.Keys.ToArray()[0];
        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

        using (var results = session.Run(inputs))
        {
            //Postprocessing
            var resultsArray = results.ToArray();
            //Pred
            var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
            var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
            //Label
            var label_value = resultsArray[1].AsEnumerable<Int64>().ToArray();

            //Fillter by score
            var candidate = GetCandidate(pred_value, pred_dim);

            if (candidate.Count != 0)
            {
                
            }
        }
        var byteArray = new byte[image.Length * 3];
        Buffer.BlockCopy(image, 0, byteArray, 0, byteArray.Length);
        var texture2d = new Texture2D(imgW, imgH, TextureFormat.RGBA32, false);
        texture2d.LoadRawTextureData(byteArray);
        texture2d.Apply();
        rawimage.texture = texture2d;
    }
    private static List<List<float>> GetCandidate(float[] pred, int[] pred_dim, float pred_thresh = 0.5f)
    {
        List<List<float>> candidate = new List<List<float>>();
        for (int batch = 0; batch < pred_dim[0]; batch++)
        {
            for (int cand = 0; cand < pred_dim[1]; cand++)
            {
                int score = 4;//Default 4  // object ness score
                int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                int idx2 = idx1 + score;
                var value = pred[idx2];
                if (value > pred_thresh)
                {
                    List<float> tmp_value = new List<float>();
                    for (int i = 0; i < pred_dim[2]; i++)
                    {
                        int sub_idx = idx1 + i;
                        tmp_value.Add(pred[sub_idx]);
                    }
                    candidate.Add(tmp_value);
                }
            }
        }
        return candidate;
    }
}
