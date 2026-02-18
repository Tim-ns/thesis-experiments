import torch
import os
import json
import glob
import re
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoProcessor, CLIPModel, ViTImageProcessor, ViTModel
import torch.nn.functional as F
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMBOOTH_ROOT = os.path.dirname(SCRIPT_DIR)

def resolve_path(path):
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(DREAMBOOTH_ROOT, path))


OBJECT_PROMPT_TEMPLATES = {
    "identity": "A photo of {0} {1}",
    "context_home": "A photo of {0} {1} at home",
    "context_table": "A photo of {0} {1} on a table",
    "context_eiffel_tower": "A photo of {0} {1} with the Eiffel Tower in the background",
    "context_grand_canyon": "A photo of {0} {1} in the Grand Canyon",
    "context_water": "A photo of {0} {1} in water",
    "context_underwater": "A photo of {0} {1} under the water",
    "context_sand": "A photo of {0} {1} buried in the sands",
}

LIVE_SUBJECT_PROMPT_TEMPLATES = {
    "identity": "A photo of {0} {1}",
    "context_table": "A photo of {0} {1} on a table",
    "context_eiffel_tower": "A photo of {0} {1} with the Eiffel Tower in the background",
    "context_grand_canyon": "A photo of {0} {1} in the Grand Canyon",
    "context_jungle": "A photo of {0} {1} in jungle",
    "context_jewelry": "A photo of {0} {1} wearing jewelry",
    "context_sunglasses": "A photo of {0} {1} wearing red sunglasses",
}

DOG_MODEL = "dog_lora_encoder"
BACKPACK_MODEL = "backpack_lora_encoder"
CAT_MODEL = "cat_lora_encoder"
SNEAKER_MODEL = "sneaker_lora_encoder"

DEFAULT_SUBJECTS = [
    {
        "id": "backpack",
        "class": "backpack",
        "type": "object",
        "lora": f"results/training/{BACKPACK_MODEL}",
        "ref": "google_dreambooth/dataset/backpack",
    },
    {
        "id": "sneaker",
        "class": "sneaker",
        "type": "object",
        "lora": f"results/training/{SNEAKER_MODEL}",
        "ref": "google_dreambooth/dataset/colorful_sneaker",
    },
    {
        "id": "dog",
        "class": "dog",
        "type": "live",
        "lora": f"results/training/{DOG_MODEL}",
        "ref": "google_dreambooth/dataset/dog",
    },
    {
        "id": "cat",
        "class": "cat",
        "type": "live",
        "lora": f"results/training/{CAT_MODEL}",
        "ref": "google_dreambooth/dataset/cat2",
    }
]


class DreamBoothEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")        
        self.dino_model = ViTModel.from_pretrained("facebook/dino-vitb16").to(device)
        self.dino_processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")

    @torch.no_grad()
    def get_clip_text_features(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        features = self.clip_model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features[0] if isinstance(features, (list, tuple)) else features.get("pooler_output", features)
        return features / features.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def get_clip_image_features(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        features = self.clip_model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features[0] if isinstance(features, (list, tuple)) else features.get("pooler_output", features)
        return features / features.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def get_dino_features(self, image):
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.dino_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :] # CLS token
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def calculate_metrics(self, gen_img, reference_images_dir, prompt):
        ref_img_paths = glob.glob(os.path.join(reference_images_dir, "*.[jJ][pP][gG]")) + \
                        glob.glob(os.path.join(reference_images_dir, "*.[pP][nN][gG]"))
        
        if not ref_img_paths:
            print(f"  Warning: No reference images found in {reference_images_dir}")
            return 0.0, 0.0, 0.0

        # Remove 'sks' (unique token) for CLIP text evaluation
        clean_prompt = re.sub(r'\bsks\b', '', prompt, flags=re.IGNORECASE).strip()
        clean_prompt = re.sub(r'  +', ' ', clean_prompt)
        
        text_feat = self.get_clip_text_features(clean_prompt)
        img_clip_feat = self.get_clip_image_features(gen_img)
        clip_t = F.cosine_similarity(text_feat, img_clip_feat).item()

        gen_dino = self.get_dino_features(gen_img)
        
        dino_scores = []
        clip_i_scores = []
        
        for ref_path in ref_img_paths:
            ref_img = Image.open(ref_path).convert("RGB")
            
            ref_dino = self.get_dino_features(ref_img)
            dino_scores.append(F.cosine_similarity(gen_dino, ref_dino).item())
            
            ref_clip = self.get_clip_image_features(ref_img)
            clip_i_scores.append(F.cosine_similarity(img_clip_feat, ref_clip).item())
        
        dino_i = sum(dino_scores) / len(dino_scores) if dino_scores else 0
        clip_i = sum(clip_i_scores) / len(clip_i_scores) if clip_i_scores else 0
        
        return clip_t, clip_i, dino_i

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    parser = argparse.ArgumentParser(description="DreamBooth Inference Pipeline")
    parser.add_argument("--subjects", type=str, help="JSON string representing subjects to add or override.")
    parser.add_argument("--ids", type=str, nargs="+", help="Specific subject IDs to run (filters the subjects list).")
    parser.add_argument("--dry_run", action="store_true", help="Print the subjects to be processed and exit.")
    parser.add_argument("--results_dir", type=str, default="results/inference", help="Main output directory for results.")
    
    args = parser.parse_args()

    subjects_map = {s["id"]: s for s in DEFAULT_SUBJECTS}

    if args.subjects:
        try:
            custom_subjects = json.loads(args.subjects)
            if not isinstance(custom_subjects, list):
                custom_subjects = [custom_subjects]
            
            for s in custom_subjects:
                if "id" not in s:
                    print(f"Warning: Subject missing 'id', skipping: {s}")
                    continue
                if s["id"] in subjects_map:
                    subjects_map[s["id"]].update(s)
                else:
                    subjects_map[s["id"]] = s
        except json.JSONDecodeError as e:
            print(f"Error parsing --subjects JSON: {e}")
            return

    # Filter by IDs if specified
    if args.ids:
        subjects_to_run = [subjects_map[sid] for sid in args.ids if sid in subjects_map]
        missing_ids = set(args.ids) - set(subjects_map.keys())
        if missing_ids:
            print(f"Warning: IDs not found in subjects: {missing_ids}")
    else:
        subjects_to_run = list(subjects_map.values())

    if not subjects_to_run:
        print("No subjects to process. Exiting.")
        return

    if args.dry_run:
        print("\nDry Run: Subjects to be processed:")
        for s in subjects_to_run:
            print(f"  - ID: {s['id']}, Class: {s['class']}, LoRA: {resolve_path(s['lora'])}")
        return

    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    evaluator = DreamBoothEvaluator(device=device)

    results_metadata = []

    for subject in subjects_to_run:
        print(f"\n{'='*50}")
        print(f"Processing Subject: {subject['id']}")
        print(f"{'='*50}")

        lora_path = resolve_path(subject["lora"])
        ref_path = resolve_path(subject["ref"])
        
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA path {lora_path} not found. Skipping.")
            continue

        pipe.load_lora_weights(lora_path)

        subject_results = {
            "subject": subject["id"],
            "reference_images_path": subject["ref"],
            "metadata": []
        }

        templates = OBJECT_PROMPT_TEMPLATES if subject["type"] == "object" else LIVE_SUBJECT_PROMPT_TEMPLATES
        
        for key, template in templates.items():
            
            prompt = template.format("sks", subject["class"])
            output_filename = f"{key}.png"
            output_path = os.path.join(args.results_dir, os.path.basename(lora_path), output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"  Generating: {output_filename} (Prompt: '{prompt}')")
            with torch.autocast(device):
                image = pipe(prompt, num_inference_steps=25).images[0]
            
            image.save(output_path)

            print(f"    Evaluating metrics...")
            ct, ci, di = evaluator.calculate_metrics(image, ref_path, prompt)

            subject_results["metadata"].append({
                "lora": lora_path,
                "prompt": prompt,
                "out": output_path,
                "clip_t": ct,
                "clip_i": ci,
                "dino_i": di
            })

        results_metadata.append(subject_results)

    metadata_out_path = os.path.join(args.results_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_out_path), exist_ok=True)
    with open(metadata_out_path, "w") as f:
        json.dump(results_metadata, f, indent=4)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Complete! Metadata saved to {metadata_out_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
