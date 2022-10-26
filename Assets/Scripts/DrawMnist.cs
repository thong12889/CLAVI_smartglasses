using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class DrawMnist : MonoBehaviour
{
    const int width = 28;
    const int height = 28;

    public MeshRenderer m;
    //public RawImage r;
    public Button b;
    public Text answer;

    private Texture2D texture;
    private readonly Color[] buffer = new Color[width * height];
    private readonly float[] input = new float[width * height];
    private MnistInference mnistInference;


    // Start is called before the first frame update
    void Start()
    {
        var path = Path.Combine(Application.streamingAssetsPath, "model.onnx");
        mnistInference = new MnistInference(path);

        //mnistInference = new MnistInference(Application.dataPath + "/MLModel/model.onnx");
        texture = new Texture2D(width, height);
        m.material.mainTexture = texture;
        //r.texture = texture;
        ClearBuffer();
    }

    // Update is called once per frame
    void Update()
    {
        //�N���b�N���W�𔒂��h��Ԃ�
        if (Input.GetMouseButton(0))
        {
            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out var hit, 100.0f))
            {
                var pixel = new Vector2(hit.textureCoord.x * width, hit.textureCoord.y * height);
                Draw(Vector2Int.RoundToInt(pixel));
            }

            texture.SetPixels(buffer);
            texture.Apply();
        }

        //�h��Ԃ���������1�Ƃ���float�̔z��ɑ��
        for (int i = 0; i < buffer.Length; i++)
        {
            input[i] = buffer[i].r;
        }

        //���_
        var num = mnistInference.Inference(input);
        answer.text = num.ToString();
    }

    //�S�s�N�Z���h��
    private void Draw(Vector2Int p)
    {
        DrawBuffer(p.x, p.y);
        DrawBuffer(p.x + 1, p.y);
        DrawBuffer(p.x + 1, p.y + 1);
        DrawBuffer(p.x, p.y + 1);
    }

    //�o�b�t�@�[�ɏ�������
    private void DrawBuffer(int x, int y)
    {
        if (x < 0 || width <= x || y < 0 || height <= y)
        {
            return;
        }

        buffer.SetValue(Color.white, x + width * y);
    }

    //��(0)�œh��Ԃ��ăN���A
    public void ClearBuffer()
    {
        for (int i = 0; i < buffer.Length; i++)
        {
            buffer[i] = Color.black;
        }

        texture.SetPixels(buffer);
        texture.Apply();
    }
}
