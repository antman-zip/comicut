import gradio as gr
import httpx
import time
import json
import os
from typing import List, Dict, Tuple

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
client = httpx.Client(timeout=300.0)

# --- API Wrapper Functions ---

def get_logs_from_api() -> str:
    try:
        response = client.get(f"{API_URL}/logs/")
        response.raise_for_status()
        logs = response.json()
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

def extract_facts_api() -> Tuple[str, str, dict]:
    try:
        response = client.post(f"{API_URL}/process/facts/")
        response.raise_for_status()
        data = response.json()
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

def analyze_image_api(image_path: str) -> str:
    if not image_path:
        return "이미지를 먼저 선택하세요."
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.split("/")[-1], f, "image/png")}
            response = client.post(f"{API_URL}/analyze-image/", files=files)
            response.raise_for_status()
            return response.json()["description"]
    except Exception as e:
        return f"분석 오류: {e}"

# --- Gradio UI Logic ---

def create_ui():
    head_html = '<link rel="icon" type="image/png" href="/icon.png">'
    
    custom_css = """
    #gallery-box {
        height: 1100px !important;
        overflow-y: auto !important;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    #gallery-box > .wrapper,
    #gallery-box > div,
    #gallery-box .grid-wrap {
        height: auto !important;
        min-height: 100% !important;
        max-height: none !important;
        overflow: visible !important;
        display: block !important;
    }
    #gallery-box .gallery-item {
        height: auto !important;
        min-height: 500px !important;
    }
    #gallery-box img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
    }
    """

    with gr.Blocks(title="ComiCut AI", theme=gr.themes.Soft(), head=head_html, css=custom_css) as demo:
        gr.Markdown("## 🎨 ComiCut AI: 리포트 → 웹툰 자동 변환")
        
        with gr.Row():
            # --- LEFT COLUMN (Controls) ---
            with gr.Column(scale=1, variant="panel"):
                # Step 1: Upload
                with gr.Group():
                    gr.Markdown("### 1. 📄 리포트 업로드")
                    report_file = gr.File(label="PDF 또는 텍스트 파일 선택", file_types=[".pdf", ".txt", ".md"])
                    with gr.Row():
                        upload_status = gr.Textbox(label="상태", interactive=False, show_label=False, container=False, scale=3)
                        btn_oneclick = gr.Button("⚡ 원클릭 실행 (1~4단계)", variant="secondary", scale=2)

                # Step 2: Facts
                with gr.Accordion("2. 🔍 핵심 팩트 추출 (FactBank)", open=False) as step2_acc:
                    btn_extract = gr.Button("팩트 추출 실행")
                    fact_status_md = gr.Markdown(value="대기 중...")
                    with gr.Accordion("📄 상세 데이터 보기 (JSON)", open=False):
                        fact_editor = gr.Code(language="json", label="FactBank JSON", lines=15)
                        btn_save_facts = gr.Button("수정사항 저장", size="sm")
                        save_status = gr.Markdown()
                    btn_extract.click(extract_facts_api, outputs=[fact_editor, fact_status_md, step2_acc])
                    btn_save_facts.click(update_facts_api, inputs=fact_editor, outputs=save_status)

                # Step 3: Draft
                with gr.Accordion("3. ✍️ 시나리오 초안 작성", open=False) as step3_acc:
                    btn_draft = gr.Button("초안 생성 실행")
                    draft_status = gr.Markdown()
                    draft_editor = gr.Textbox(lines=15, label="시나리오 초안", interactive=True)
                    btn_save_draft = gr.Button("수정사항 저장", size="sm")
                    btn_draft.click(generate_draft_api, outputs=[draft_editor, draft_status])
                    btn_save_draft.click(update_draft_api, inputs=draft_editor, outputs=draft_status)

                # Step 4: Final Refine
                with gr.Accordion("4. ✅ 최종 검증 및 정제", open=False) as step4_acc:
                    btn_refine = gr.Button("검증 및 정제 실행 (이미지용)")
                    refine_status = gr.Markdown()
                    final_editor = gr.Textbox(lines=15, label="최종 시나리오 (이미지 프롬프트)", interactive=True)
                    with gr.Row():
                        btn_save_final = gr.Button("최종본 저장", size="sm")
                        btn_download_scenario = gr.DownloadButton("💾 시나리오 다운로드 (.txt)", size="sm")
                    btn_refine.click(refine_scenario_api, outputs=[final_editor, refine_status])
                    btn_save_final.click(update_final_api, inputs=final_editor, outputs=refine_status)
                    
                    def download_scenario_func(text):
                        if not text: return None
                        tmp_dir = os.path.join(os.getcwd(), "tmp")
                        os.makedirs(tmp_dir, exist_ok=True)
                        filename = os.path.join(tmp_dir, f"scenario_{int(time.time())}.txt")
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(text)
                        return filename
                    
                    btn_download_scenario.click(download_scenario_func, inputs=final_editor, outputs=btn_download_scenario)

                # Step 5: Generation
                with gr.Group():
                    gr.Markdown("### 5. 🎨 이미지 생성")
                    char_inputs = []
                    with gr.Accordion("캐릭터 설정 (선택사항)", open=False):
                        for i in range(3):
                            with gr.Tab(f"캐릭터 {i+1}"):
                                c_name = gr.Textbox(label="이름")
                                c_enable = gr.Checkbox(label="활성화", value=True)
                                with gr.Row():
                                    c_img = gr.Image(label="참조 이미지", type="filepath", height=300)
                                    with gr.Column():
                                        c_desc = gr.Textbox(label="외관 묘사 (Reference Prompt)", placeholder="예: 파란 정장, 단발머리", lines=5)
                                        btn_analyze = gr.Button("🔍 AI 이미지 분석 및 입력", size="sm")
                                        btn_analyze.click(analyze_image_api, inputs=[c_img], outputs=[c_desc])
                                char_inputs.extend([c_name, c_enable, c_img, c_desc])
                    
                    resolution = gr.Radio(["1K", "2K", "4K"], label="해상도", value="1K")
                    btn_start = gr.Button("✨ 웹툰 생성 시작", variant="primary")
                    gen_msg = gr.Markdown()

            # --- RIGHT COLUMN (Gallery & Logs) ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🖼️ 생성 결과 (갤러리)"):
                        gallery = gr.Gallery(
                            label="생성된 컷", 
                            columns=2, 
                            object_fit="contain",
                            interactive=False,
                            elem_id="gallery-box"
                        )
                        btn_download = gr.DownloadButton("💾 전체 이미지 ZIP 다운로드", visible=True)
                        
                    with gr.Tab("📝 시스템 로그"):
                        logs = gr.Textbox(lines=30, label="로그", interactive=False, autoscroll=True)

        # --- Event Wiring ---
        report_file.upload(lambda f: upload_report_to_api(f) if f else "파일 없음", inputs=report_file, outputs=upload_status)
        
        # ... (pipeline functions omitted for brevity) ...

        # Download ZIP Logic
        def download_zip_func():
            try:
                # Request backend to create ZIP and return path
                resp = client.post(f"{API_URL}/create-zip/")
                resp.raise_for_status()
                return resp.json()["path"]
            except Exception as e:
                return None

        btn_download.click(download_zip_func, outputs=btn_download)

        def run_pipeline_all(file_path):
            if not file_path:
                yield "❌ 파일을 먼저 업로드하세요.", *([gr.update()]*8)
                return
            yield "🚀 원클릭 실행 시작...", *([gr.update()]*8)
            
            yield "🔍 팩트 추출 중...", *([gr.update()]*8)
            facts_json, fact_msg, _ = extract_facts_api()
            if "실패" in fact_msg:
                yield f"❌ 2단계 실패: {fact_msg}", *([gr.update()]*8)
                return
            yield "✅ 팩트 추출 완료. 시나리오 작성 중...", facts_json, fact_msg, gr.update(open=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            draft_text, draft_msg = generate_draft_api()
            if "실패" in draft_msg:
                yield f"❌ 3단계 실패: {draft_msg}", facts_json, fact_msg, gr.update(open=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                return
            yield "✅ 초안 작성 완료. 검증 중...", facts_json, fact_msg, gr.update(open=True), draft_text, draft_msg, gr.update(open=True), gr.update(), gr.update()
            
            final_text, refine_msg = refine_scenario_api()
            if "실패" in refine_msg:
                yield f"❌ 4단계 실패: {refine_msg}", facts_json, fact_msg, gr.update(open=True), draft_text, draft_msg, gr.update(open=True), gr.update(), gr.update()
                return
            yield "🎉 1~4단계 원클릭 실행 완료!", facts_json, fact_msg, gr.update(open=True), draft_text, draft_msg, gr.update(open=True), final_text, refine_msg

        btn_oneclick.click(
            run_pipeline_all,
            inputs=[report_file],
            outputs=[upload_status, fact_editor, fact_status_md, step2_acc, draft_editor, draft_status, step3_acc, final_editor, refine_status]
        )

        def run_generation(res, final_scenario_text, *chars):
            try:
                all_chars = client.get(f"{API_URL}/characters/").json()
                for c in all_chars: client.delete(f"{API_URL}/characters/{c['id']}")
                for i in range(3):
                    base_idx = i * 4
                    name, en, img, desc = chars[base_idx], chars[base_idx+1], chars[base_idx+2], chars[base_idx+3]
                    if name:
                        r = client.post(f"{API_URL}/characters/")
                        cid = r.json()['id']
                        payload = {"name": name, "enabled": en, "image": None, "description": desc}
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
            
            last_img_count = -1
            while True:
                l = get_logs_from_api()
                p = get_panels_from_api()
                
                imgs = []
                working = False
                current_img_count = 0
                for pan in p:
                    if pan['imageUrl']:
                        imgs.append((pan['imageUrl'], f"Cut {pan['cutNumber']}"))
                        current_img_count += 1
                    if pan['status'] in ['pending', 'generating']:
                        working = True
                
                if current_img_count != last_img_count or not working:
                    yield msg, imgs, l
                    last_img_count = current_img_count
                
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
        
        demo.load(get_logs_from_api, outputs=logs)

    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch()