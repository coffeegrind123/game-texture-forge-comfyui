.PHONY: build up down logs shell ps restart nodes models gpu-test

build:        ## Build the image
	docker compose build

up:           ## Start ComfyUI (detached) -> http://localhost:8188
	docker compose up -d

down:         ## Stop and remove the container
	docker compose down

restart:      ## Restart the container
	docker compose restart

logs:         ## Follow container logs
	docker compose logs -f

shell:        ## Open a shell inside the running container
	docker compose exec comfyui bash

ps:           ## Show container status
	docker compose ps

nodes:        ## List installed custom nodes
	docker compose exec comfyui ls -1 /opt/ComfyUI/custom_nodes

models:       ## Download starter models into the container. CHORD is gated: make models HF_TOKEN=hf_xxx
	docker compose exec -e HF_TOKEN=$(HF_TOKEN) -T comfyui bash -s < scripts/download_models.sh

workflow:     ## (Re)install the generated workflow into the container's sidebar
	docker compose cp workflows/texture_img2img_tiling.json \
	  comfyui:/opt/ComfyUI/user/default/workflows/texture_img2img_tiling.json

put-input:    ## Copy a local image into the container input dir:  make put-input IMG=path/to/tex.png
	docker compose cp "$(IMG)" comfyui:/opt/ComfyUI/input/

get-output:   ## Copy generated outputs out of the container to ./out
	mkdir -p out && docker compose cp comfyui:/opt/ComfyUI/output/. out/

gpu-test:     ## Verify the container can see the GPU via torch
	docker compose run --rm --entrypoint python comfyui -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
