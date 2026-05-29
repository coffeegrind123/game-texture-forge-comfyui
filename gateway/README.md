# Texture Pipeline Gateway

A high-level REST API over ComfyUI so AI agents (and your s&box engine) can drive
the texture pipeline without constructing node graphs. Fronts the `comfyui`
service; talks to it purely over HTTP/WS (no shared volumes).

- **Swagger UI:** `http://<host>:8080/docs`
- **OpenAPI schema:** `http://<host>:8080/openapi.json` (use for client codegen)
- CORS is open (`*`) so browser / engine clients can call it directly.

## Operations

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | gateway + ComfyUI reachability |
| GET  | `/capabilities` | available checkpoints, controlnets, upscalers, samplers, CHORD flag |
| POST | `/restyle` | existing texture → realistic variant, tiling preserved/created |
| POST | `/make-seamless` | make a non-tiling image tile (radial/halfshift/mirrored, no diffusion) |
| POST | `/pbr` | CHORD → BaseColor/Normal/Roughness/Metalness |
| GET  | `/jobs/{id}` | poll status / progress (0..1) / outputs |
| GET  | `/jobs/{id}/outputs/{label}` | download a result PNG |

All three operations are **async**: they return a `job_id` immediately; poll
`/jobs/{id}` until `status` is `completed` (or `failed`), then download each
output. Labels: `restyle`/`make-seamless` → `result`; `pbr` →
`basecolor`,`normal`,`roughness`,`metalness`.

## Supplying the input image

Any one of:
- **multipart**: form fields `file` (the image) + `params` (a JSON string)
- **JSON** body with `image_base64` (data-URI prefix optional)
- **JSON** body with `image_url` (gateway fetches it)

## Examples

Restyle via JSON + base64, then poll and download:
```bash
HOST=http://localhost:8080
B64=$(base64 -w0 tex.png)
JOB=$(curl -s $HOST/restyle -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$B64\",\"denoise\":0.45,\"tiling\":\"enable\",\"upscale\":true}" \
  | jq -r .job_id)
# poll
curl -s $HOST/jobs/$JOB | jq '{status,progress}'
# download when completed
curl -s $HOST/jobs/$JOB/outputs/result -o out.png
```

Restyle via multipart file upload:
```bash
curl -s $HOST/restyle \
  -F file=@tex.png \
  -F 'params={"denoise":0.5,"prompt":"weathered granite, photoreal"}'
```

PBR maps:
```bash
curl -s $HOST/pbr -F file=@tex.png            # -> job with 4 outputs
curl -s $HOST/jobs/$JOB/outputs/normal -o normal.png
```

## Key `/restyle` params (see `/docs` for all)

| param | default | meaning |
|---|---|---|
| `denoise` | 0.45 | lower = closer to source, higher = more new detail |
| `tiling` | `enable` | `enable`/`x_only`/`y_only`/`disable` (circular model+VAE) |
| `use_controlnet` / `controlnet_strength` | true / 0.6 | Tile-CN structure lock |
| `upscale` / `upscale_pad` | true / 64 | tiling-aware model upscale |
| `input_size` | 1024 | resize longest side (aspect-preserved); `null` = as-is |
| `prompt` / `negative_prompt` | photoreal defaults | text guidance |
| `seed` | -1 (random) | reproducibility |
| `steps` / `cfg` / `sampler_name` / `scheduler` | 28 / 6.5 / dpmpp_2m / karras | sampler |

## s&box integration (`~/sbox-public`)

s&box sandboxes networking through `Sandbox.Http`. Point it at the gateway host.
This sends a texture, polls, and downloads the restyled result:

```csharp
using Sandbox;
using System.Text.Json;
using System.Threading.Tasks;

public static class TextureGateway
{
    const string Host = "http://localhost:8080"; // your gateway host

    public static async Task<byte[]> RestyleAsync( byte[] png, float denoise = 0.45f )
    {
        var b64 = System.Convert.ToBase64String( png );
        var body = JsonSerializer.Serialize( new {
            image_base64 = b64, denoise, tiling = "enable", upscale = true
        } );

        // submit
        var submit = await Http.RequestJsonAsync<JobDto>(
            $"{Host}/restyle", "POST",
            new StringContent( body, System.Text.Encoding.UTF8, "application/json" ) );

        // poll
        JobDto job;
        do
        {
            await Task.DelayRealtimeSeconds( 2f );
            job = await Http.RequestJsonAsync<JobDto>( $"{Host}/jobs/{submit.job_id}" );
        }
        while ( job.status is "queued" or "running" );

        if ( job.status != "completed" )
            throw new System.Exception( $"texture job failed: {job.error}" );

        // download the result PNG
        return await Http.RequestBytesAsync( $"{Host}/jobs/{submit.job_id}/outputs/result" );
        // -> Texture.Load / create a runtime texture from these bytes
    }

    public class JobDto
    {
        public string job_id { get; set; }
        public string status { get; set; }
        public float progress { get; set; }
        public string error { get; set; }
    }
}
```

Notes:
- Exact `Sandbox.Http` method names vary by s&box version; adapt to whatever your
  build exposes (`Http.RequestJsonAsync` / `RequestStringAsync` / `RequestBytesAsync`).
  The shapes (POST JSON, GET JSON, GET bytes) are what matter.
- Published games may need the gateway host whitelisted for network access; in the
  editor/tools context it's unrestricted.
- For a fully-typed client, generate one from `/openapi.json`.

## Running

Built and started by the project compose: `docker compose up -d gateway`.
Rebuild after code changes: `docker compose build gateway && docker compose up -d gateway`.
Env: `COMFY_URL` (default `http://comfyui:8188`).

> Jobs are tracked in memory — fine for single-instance use. Restarting the
> gateway clears job history (outputs still live in ComfyUI's `output/`).
