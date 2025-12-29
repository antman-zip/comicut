import gradio as gr
import httpx
import time
import uuid
import json
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
client = httpx.Client(timeout=300.0) # Increased timeout for LLM tasks

# --- API Wrapper Functions ---

def get_logs_from_api() -> str:
    try:
        response = client.get(f"{API_URL}/logs/")
        response.raise_for_status()
        logs = response.json()
        # Sort logs by timestamp desc (newest first)
        logs.reverse()
        log_messages = [f"[{log['timestamp']}] {log['message']}" for log in logs]
        return "\n".join(log_messages)
    except Exception:
        return "Waiting for logs..."

def get_panels_from_api() -> List[Dict]:
    try:
        response = client.get(f"{API_URL}/panels/")
        response.raise_for_status()
        return response.json()
    except Exception:
        return []

# Step 1: Upload
def upload_report_to_api(file_path: str) -> str:
    if not file_path:
        raise ValueError("파일이 없습니다.")
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.split("/")[-1], f, "application/pdf" if file_path.lower().endswith('.pdf') else "text/plain")}
            response = client.post(f"{API_URL}/upload-report/", files=files)
            response.raise_for_status()
            return f"✅ 업로드 완료: {response.json()['filename']}"
    except Exception as e:
        raise RuntimeError(f"업로드 실패: {e}")

# Step 2: Facts
def extract_facts_api() -> Tuple[str, str, dict]:
    try:
        response = client.post(f"{API_URL}/process/facts/")
        response.raise_for_status()
        data = response.json()
        
        # 요약 정보 생성
        fact_count = len(data.get("facts", []))
        page_count = data.get("factbank_meta", {}).get("page_count", "?")
        summary_msg = f"### ✅ 추출 완료\n- **총 팩트 수**: {fact_count}개\n- **페이지 수**: {page_count}쪽"
        
        return json.dumps(data, indent=2, ensure_ascii=False), summary_msg, gr.update(open=True)
    except Exception as e:
        return "{}", f"❌ 추출 실패: {e}", gr.update(open=True)

def update_facts_api(json_text: str) -> str:
    try:
        data = json.loads(json_text)
        client.put(f"{API_URL}/pipeline/facts/", json={"facts": data})
        return "✅ 팩트 수정사항이 저장되었습니다."
    except Exception as e:
        return f"❌ 저장 실패: {e}"

# Step 3: Draft
def generate_draft_api() -> Tuple[str, str]:
    try:
        response = client.post(f"{API_URL}/process/draft/")
        response.raise_for_status()
        return response.json()["draft"], "✅ 시나리오 초안이 생성되었습니다."
    except Exception as e:
        return "", f"❌ 초안 생성 실패: {e}"

def update_draft_api(text: str) -> str:
    try:
        client.put(f"{API_URL}/pipeline/draft/", json={"scenario_text": text})
        return "✅ 초안 수정사항이 저장되었습니다."
    except Exception as e:
        return f"❌ 저장 실패: {e}"

# Step 4: Refine
def refine_scenario_api() -> Tuple[str, str]:
    try:
        response = client.post(f"{API_URL}/process/refine/")
        response.raise_for_status()
        return response.json()["final_scenario"], "✅ 시나리오 검증 및 정제 완료."
    except Exception as e:
        return "", f"❌ 정제 실패: {e}"

def update_final_api(text: str) -> str:
    try:
        client.put(f"{API_URL}/pipeline/final/", json={"scenario_text": text})
        return "✅ 최종 시나리오가 저장되었습니다."
    except Exception as e:
        return f"❌ 저장 실패: {e}"

# Step 5: Start Gen
def start_generation_on_api(script_text: str = None) -> str:
    try:
        payload = {}
        if script_text and script_text.strip():
            payload = {"script_text": script_text}
            
        response = client.post(f"{API_URL}/generate/start/", json=payload)
        response.raise_for_status()
        return "🚀 이미지 생성이 시작되었습니다."
    except Exception as e:
        return f"❌ 생성 시작 오류: {e}"

# --- Gradio UI Logic ---

def create_ui():
    # Custom Head for Favicon
    # Ensure 'icon.png' exists in the project root folder
    head_html = """
    <link rel="icon" type="image/png" href="/icon.png">
    """

    with gr.Blocks(title="ComiCut AI", theme=gr.themes.Soft(), head=head_html) as demo:
        gr.Markdown("## 🎨 ComiCut AI: 리포트 → 웹툰 자동 변환")
        
        with gr.Row():
            # --- LEFT COLUMN: Pipeline Control ---
            with gr.Column(scale=1, variant="panel"):
                
                # Step 1
                with gr.Group():
                    gr.Markdown("### 1. 📄 리포트 업로드")
                    report_file = gr.File(label="PDF 또는 텍스트 파일 선택", file_types=[".pdf", ".txt", ".md"])
                    
                    with gr.Row():
                        upload_status = gr.Textbox(label="상태", interactive=False, show_label=False, container=False, scale=3)
                        btn_oneclick = gr.Button("⚡ 원클릭 실행 (1~4단계)", variant="secondary", scale=2)

                # Step 2
                with gr.Accordion("2. 🔍 핵심 팩트 추출 (FactBank)", open=False) as step2_acc:
                    btn_extract = gr.Button("팩트 추출 실행")
                    fact_status_md = gr.Markdown(value="대기 중...")
                    with gr.Accordion("📄 상세 데이터 보기 (JSON)", open=False):
                        fact_editor = gr.Code(language="json", label="FactBank JSON", lines=15)
                        btn_save_facts = gr.Button("수정사항 저장", size="sm")
                        save_status = gr.Markdown()

                    btn_extract.click(
                        extract_facts_api, 
                        outputs=[fact_editor, fact_status_md, step2_acc]
                    )
                    btn_save_facts.click(update_facts_api, inputs=fact_editor, outputs=save_status)

                # Step 3
                with gr.Accordion("3. ✍️ 시나리오 초안 작성", open=False) as step3_acc:
                    btn_draft = gr.Button("초안 생성 실행")
                    draft_status = gr.Markdown()
                    draft_editor = gr.Textbox(lines=15, label="시나리오 초안", interactive=True)
                    btn_save_draft = gr.Button("수정사항 저장", size="sm")
                    
                    btn_draft.click(generate_draft_api, outputs=[draft_editor, draft_status])
                    btn_save_draft.click(update_draft_api, inputs=draft_editor, outputs=draft_status)

                # Step 4
                with gr.Accordion("4. ✅ 최종 검증 및 정제", open=False) as step4_acc:
                    btn_refine = gr.Button("검증 및 정제 실행 (이미지용)")
                    refine_status = gr.Markdown()
                    final_editor = gr.Textbox(lines=15, label="최종 시나리오 (이미지 프롬프트)", interactive=True)
                    btn_save_final = gr.Button("최종본 저장", size="sm")
                    
                    btn_refine.click(refine_scenario_api, outputs=[final_editor, refine_status])
                    btn_save_final.click(update_final_api, inputs=final_editor, outputs=refine_status)

                # Step 5
                with gr.Group():
                    gr.Markdown("### 5. 🎨 이미지 생성")
                    
                    # Character Slots
                    char_inputs = []
                    with gr.Accordion("캐릭터 설정 (선택사항)", open=False):
                        for i in range(3):
                            with gr.Tab(f"캐릭터 {i+1}"):
                                c_name = gr.Textbox(label="이름")
                                c_enable = gr.Checkbox(label="활성화", value=True)
                                c_img = gr.Image(label="참조 이미지", type="filepath", height=100)
                                char_inputs.extend([c_name, c_enable, c_img])
                    
                    resolution = gr.Radio(["1K", "2K", "4K"], label="해상도", value="1K")
                    btn_start = gr.Button("✨ 웹툰 생성 시작", variant="primary")
                    gen_msg = gr.Markdown()

            # --- RIGHT COLUMN: Output & Logs ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🖼️ 생성 결과 (갤러리)"):
                        # height=1200 ensures a large, scrollable area within the gallery container
                        gallery = gr.Gallery(label="생성된 컷", columns=2, height=1200)
                        # Download Button
                        btn_download = gr.Button("💾 전체 이미지 ZIP 다운로드", visible=True)
                        download_file = gr.File(label="다운로드 파일", visible=False)
                        download_status = gr.Markdown()
                        
                    with gr.Tab("📝 시스템 로그"):
                        logs = gr.Textbox(lines=30, label="로그 (프롬프트 포함)", interactive=False, autoscroll=True)

        # --- Event Wiring ---
        
        # 1. File Upload
        report_file.upload(
            lambda f: upload_report_to_api(f) if f else "파일 없음",
            inputs=report_file,
            outputs=upload_status
        )

        # 2. One-Click Pipeline (Generator for progress updates)
        def run_pipeline_all(file_path):
            if not file_path:
                yield "❌ 파일을 먼저 업로드하세요.", *([gr.update()]*8)
                return

            yield "🚀 원클릭 실행 시작...", *([gr.update()]*8)
            
            # Step 2: Facts
            yield "🔍 팩트 추출 중...", *([gr.update()]*8)
            facts_json, fact_msg, _ = extract_facts_api()
            if "실패" in fact_msg:
                yield f"❌ 2단계 실패: {fact_msg}", *([gr.update()]*8)
                return
            yield "✅ 팩트 추출 완료. 시나리오 작성 중...", facts_json, fact_msg, gr.update(open=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            # Step 3: Draft
            draft_text, draft_msg = generate_draft_api()
            if "실패" in draft_msg:
                yield f"❌ 3단계 실패: {draft_msg}", facts_json, fact_msg, gr.update(open=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                return
            yield "✅ 초안 작성 완료. 검증 중...", facts_json, fact_msg, gr.update(open=True), draft_text, draft_msg, gr.update(open=True), gr.update(), gr.update()
            
            # Step 4: Refine
            final_text, refine_msg = refine_scenario_api()
            if "실패" in refine_msg:
                yield f"❌ 4단계 실패: {refine_msg}", facts_json, fact_msg, gr.update(open=True), draft_text, draft_msg, gr.update(open=True), gr.update(), gr.update()
                return
                
            yield "🎉 1~4단계 원클릭 실행 완료!", facts_json, fact_msg, gr.update(open=True), draft_text, draft_msg, gr.update(open=True), final_text, refine_msg

        btn_oneclick.click(
            run_pipeline_all,
            inputs=[report_file],
            outputs=[
                upload_status, 
                fact_editor, fact_status_md, step2_acc,
                draft_editor, draft_status, step3_acc,
                final_editor, refine_status
            ]
        )

        # 3. Image Generation
        def run_generation(res, final_scenario_text, *chars):
            # Sync characters
            try:
                all_chars = client.get(f"{API_URL}/characters/").json()
                for c in all_chars: client.delete(f"{API_URL}/characters/{c['id']}")
                for i in range(3):
                    name, en, img = chars[i*3], chars[i*3+1], chars[i*3+2]
                    if name:
                        r = client.post(f"{API_URL}/characters/")
                        cid = r.json()['id']
                        payload = {"name": name, "enabled": en, "image": None}
                        if img:
                            import base64
                            with open(img, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            payload["image"] = f"data:image/png;base64,{b64}"
                        client.put(f"{API_URL}/characters/{cid}", json=payload)
            except Exception as e:
                yield f"❌ 캐릭터 동기화 오류: {e}", [], ""
                return

            msg = start_generation_on_api(final_scenario_text)
            yield msg, [], ""
            
            while True:
                l = get_logs_from_api()
                p = get_panels_from_api()
                
                imgs = []
                working = False
                for pan in p:
                    # Handle both URL and Base64 (legacy)
                    if pan['imageUrl']:
                        imgs.append((pan['imageUrl'], f"Cut {pan['cutNumber']}"))
                    if pan['status'] in ['pending', 'generating']:
                        working = True
                
                yield msg, imgs, l
                
                if not working and len(p) > 0:
                    break
                if "오류" in msg or "Error" in msg:
                    break
                time.sleep(2)
            
            yield "✅ 생성 완료!", imgs, get_logs_from_api()

        btn_start.click(
            run_generation,
            inputs=[resolution, final_editor] + char_inputs,
            outputs=[gen_msg, gallery, logs]
        )
        
        # 4. Download (Using create-zip endpoint to avoid SSRF)
        def download_zip_action():
            try:
                response = client.post(f"{API_URL}/create-zip/")
                response.raise_for_status()
                path = response.json()["path"]
                return path, "✅ 다운로드 준비 완료."
            except Exception as e:
                return None, f"❌ 다운로드 오류: {e}"

        btn_download.click(
            download_zip_action,
            outputs=[download_file, download_status]
        ).then(
            lambda: gr.update(visible=True), outputs=download_file
        )
        
        demo.load(get_logs_from_api, outputs=logs)

    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch()
