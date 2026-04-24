using UnityEngine;
using System.Collections.Generic;
using System.IO;

[System.Serializable]
public class HoldData
{
    public int id;
    public List<float> pos;
    public string type;
}

[System.Serializable]
public class ClimbingResult
{
    public string pipeline;
    public List<HoldData> holds;
    public List<int> path;
    public List<string> path_stances;
    public float final_height;
    public float goal_height;
}

public class ClimbingPathPlayer : MonoBehaviour
{
    [Header("설정")]
    public float moveSpeed = 1f;
    public float holdSize = 0.1f;
    public bool autoPlay = true;

    [Header("색상")]
    public Color startHoldColor = Color.red;
    public Color normalHoldColor = new Color(0.9f, 0.4f, 0.2f);
    public Color goalHoldColor = Color.green;
    public Color handColor = Color.red;
    public Color footColor = Color.green;
    public Color torsoColor = new Color(1f, 0.6f, 0.4f);

    private ClimbingResult result;
    private List<Vector3> holdPositions = new List<Vector3>();
    private List<int> path;
    private int currentIndex = 0;
    private float progress = 0;

    // Avatar parts
    private GameObject handL, handR, footL, footR;
    private GameObject pelvis, spine, chest, head;
    private GameObject shoulderL, shoulderR;
    private GameObject avatarRoot;

    // Path visualization
    private LineRenderer pathLine;

    void Start()
    {
        LoadResult();
        CreateHolds();
        CreateAvatar();
        CreatePathLine();

        if (autoPlay)
            Play();
    }

    void LoadResult()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "climbing_result_stance.json");

        if (File.Exists(path))
        {
            string json = File.ReadAllText(path);
            result = JsonUtility.FromJson<ClimbingResult>(json);

            if (result != null)
            {
                foreach (var hold in result.holds)
                {
                    holdPositions.Add(new Vector3(hold.pos[0], hold.pos[1], hold.pos[2]));
                }
                this.path = new List<int>(result.path);
                Debug.Log($"경로 로드: {this.path.Count} 스텝, {holdPositions.Count} 홀드");
            }
        }
        else
        {
            Debug.LogWarning($"파일 없음: {path}");
        }
    }

    void CreateHolds()
    {
        if (result == null || result.holds == null) return;

        for (int i = 0; i < result.holds.Count; i++)
        {
            var hold = result.holds[i];
            GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            obj.transform.position = holdPositions[i];
            obj.transform.localScale = Vector3.one * holdSize;
            obj.name = $"Hold_{hold.id}";

            Color color = normalHoldColor;
            if (hold.type == "start") color = startHoldColor;
            else if (hold.type == "goal") color = goalHoldColor;

            obj.GetComponent<Renderer>().material.color = color;
        }
    }

    void CreateAvatar()
    {
        avatarRoot = new GameObject("Avatar");
        avatarRoot.transform.position = new Vector3(2f, 0.5f, 0);

        // 몸통
        pelvis = CreatePart("Pelvis", new Vector3(0.2f, 0.12f, 0.12f), new Vector3(0, 0, 0), torsoColor, avatarRoot);
        spine = CreatePart("Spine", new Vector3(0.18f, 0.2f, 0.12f), new Vector3(0, 0.2f, 0), torsoColor, avatarRoot);
        chest = CreatePart("Chest", new Vector3(0.22f, 0.18f, 0.14f), new Vector3(0, 0.4f, 0), torsoColor, avatarRoot);
        head = CreatePart("Head", new Vector3(0.12f, 0.14f, 0.12f), new Vector3(0, 0.6f, 0), torsoColor, avatarRoot);

        // 어깨
        shoulderL = CreatePart("ShoulderL", new Vector3(0.08f, 0.08f, 0.08f), new Vector3(-0.2f, 0.5f, 0), torsoColor, avatarRoot);
        shoulderR = CreatePart("ShoulderR", new Vector3(0.08f, 0.08f, 0.08f), new Vector3(0.2f, 0.5f, 0), torsoColor, avatarRoot);

        // 손
        handL = CreatePart("HandL", new Vector3(0.06f, 0.1f, 0.06f), new Vector3(-0.3f, 0.4f, 0), handColor, avatarRoot);
        handR = CreatePart("HandR", new Vector3(0.06f, 0.1f, 0.06f), new Vector3(0.3f, 0.4f, 0), handColor, avatarRoot);

        // 발
        footL = CreatePart("FootL", new Vector3(0.06f, 0.08f, 0.1f), new Vector3(-0.1f, -0.2f, 0), footColor, avatarRoot);
        footR = CreatePart("FootR", new Vector3(0.06f, 0.08f, 0.1f), new Vector3(0.1f, -0.2f, 0), footColor, avatarRoot);
    }

    GameObject CreatePart(string name, Vector3 scale, Vector3 localPos, Color color, GameObject parent)
    {
        GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        obj.name = name;
        obj.transform.parent = parent.transform;
        obj.transform.localPosition = localPos;
        obj.transform.localScale = scale;
        obj.GetComponent<Renderer>().material.color = color;
        return obj;
    }

    void CreatePathLine()
    {
        if (pathLine != null || path == null || holdPositions == null) return;

        pathLine = avatarRoot.AddComponent<LineRenderer>();
        pathLine.positionCount = path.Count;
        pathLine.startWidth = 0.02f;
        pathLine.endWidth = 0.02f;
        pathLine.material = new Material(Shader.Find("Sprites/Default"));
        pathLine.startColor = Color.yellow;
        pathLine.endColor = Color.yellow;

        for (int i = 0; i < path.Count; i++)
        {
            if (path[i] >= 0 && path[i] < holdPositions.Count)
            {
                Vector3 pos = holdPositions[path[i]];
                pathLine.SetPosition(i, pos + Vector3.up * 0.05f);
            }
        }
    }

    public void Play()
    {
        if (path == null || path.Count == 0)
        {
            Debug.LogWarning("경로 없음");
            return;
        }

        currentIndex = 0;
        progress = 0;
        Debug.Log($"등반 시작! {path.Count} 스텝");
    }

    void Update()
    {
        if (path == null || currentIndex >= path.Count - 1) return;

        progress += Time.deltaTime * moveSpeed;

        if (progress >= 1f)
        {
            progress = 0;
            currentIndex++;

            if (currentIndex >= path.Count - 1)
            {
                Debug.Log("등반 완료!");
                return;
            }
        }

        UpdateAvatarPosition();
    }

    void UpdateAvatarPosition()
    {
        if (holdPositions == null) return;

        int idx1 = path[currentIndex];
        int idx2 = path[Mathf.Min(currentIndex + 1, path.Count - 1)];

        if (idx1 < 0 || idx1 >= holdPositions.Count || idx2 < 0 || idx2 >= holdPositions.Count) return;

        Vector3 pos1 = holdPositions[idx1];
        Vector3 pos2 = holdPositions[idx2];

        float t = Mathf.SmoothStep(0, 1, progress);
        Vector3 handCenter = Vector3.Lerp(pos1, pos2, t);

        // 손 위치 (交互)
        if (currentIndex % 2 == 0)
        {
            if (handL != null) handL.transform.position = handCenter;
            if (handR != null) handR.transform.position = handCenter + new Vector3(0.2f, 0, 0);
        }
        else
        {
            if (handL != null) handL.transform.position = handCenter + new Vector3(-0.2f, 0, 0);
            if (handR != null) handR.transform.position = handCenter;
        }

        // 몸통 위치
        Vector3 bodyOffset = new Vector3(0, -0.35f, -0.1f);
        Vector3 bodyPos = handCenter + bodyOffset;

        if (pelvis != null) pelvis.transform.position = bodyPos;
        if (spine != null) spine.transform.position = bodyPos + new Vector3(0, 0.2f, 0);
        if (chest != null) chest.transform.position = bodyPos + new Vector3(0, 0.4f, 0);
        if (head != null) head.transform.position = bodyPos + new Vector3(0, 0.65f, 0);

        // 어깨
        if (shoulderL != null) shoulderL.transform.position = bodyPos + new Vector3(-0.2f, 0.45f, 0);
        if (shoulderR != null) shoulderR.transform.position = bodyPos + new Vector3(0.2f, 0.45f, 0);

        // 발 (아래)
        if (footL != null) footL.transform.position = handCenter + new Vector3(-0.15f, -0.4f, 0);
        if (footR != null) footR.transform.position = handCenter + new Vector3(0.15f, -0.4f, 0);
    }
}