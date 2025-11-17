# app.py — Ethical Crossroads: African Context Edition
# author: Prof. Songhee Kang
# AIM 2025, Fall. TU Korea

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== App Config ====================
st.set_page_config(page_title="윤리적 전환: 아프리카 컨텍스트", page_icon="🌍", layout="centered")

# ==================== Global Timeout ====================
HTTPX_TIMEOUT = httpx.Timeout(
    connect=15.0, read=180.0, write=30.0, pool=15.0
)

# ==================== Utils ====================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def coerce_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("JSON 블록을 찾지 못했습니다.")
    js = m.group(0)
    js = re.sub(r",\s*([\]}])", r"\1", js)
    return json.loads(js)

def get_secret(k: str, default: str=""):
    try:
        return st.secrets.get(k, os.getenv(k, default))
    except Exception:
        return os.getenv(k, default)

# ==================== DNA Client ====================
def _render_chat_template_str(messages: List[Dict[str,str]]) -> str:
    def block(role, content): return f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"
    sys = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            sys = block("system", m["content"])
        else:
            rest.append(block(m["role"], m["content"]))
    return sys + "".join(rest) + "\n<|im_start|>assistant<|im_sep|>"

class DNAClient:
    def __init__(self, backend: str, model_id: str, api_key: Optional[str], endpoint_url: Optional[str], api_key_header: str, temperature: float):
        self.backend = backend
        self.model_id = model_id
        self.api_key = api_key or get_secret("HF_TOKEN")
        self.endpoint_url = endpoint_url or get_secret("DNA_R1_ENDPOINT", "http://210.93.49.11:8081/v1")
        self.temperature = temperature
        self.api_key_header = api_key_header

    def _auth_headers(self) -> Dict[str,str]:
        h = {"Content-Type":"application/json"}
        if not self.api_key: return h
        hk = self.api_key_header.strip().lower()
        if hk.startswith("authorization"): h["Authorization"] = f"Bearer {self.api_key}"
        elif hk in {"api-key", "x-api-key"}: h["API-KEY"] = self.api_key
        else: h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 900) -> str:
        if self.backend == "openai":
            url = self.endpoint_url.rstrip("/") + "/chat/completions"
            payload = {
                "messages": messages, "temperature": self.temperature, "max_tokens": max_new_tokens, "stream": False
            }
            if self.model_id: payload["model"] = self.model_id
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        elif self.backend == "tgi":
            url = self.endpoint_url.rstrip("/") + "/generate"
            prompt = _render_chat_template_str(messages)
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_new_tokens, "temperature": self.temperature, "stop": ["<|im_end|>"]},
                "stream": False
            }
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data.get("generated_text") if isinstance(data, dict) else data[0].get("generated_text", "")
        else:
            # Fallback or Local placeholder
            return "{}"

# ==================== Scenario Model (African Context) ====================
@dataclass
class SubOption:
    framework: str  # emotion, social, identity, moral
    description: str
    rationale: str

@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    main_options: Dict[str, str]  # {"A": "...", "B": "..."}
    sub_options: Dict[str, List[SubOption]] # {"A": [SubOption...], "B": [SubOption...]}
    base_stats: Dict[str, Dict[str, float]] # Basic stats for A vs B

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 1. Refugee Scenario
s1_sub_a = [
    SubOption("emotion", "난민들에게 가장 따뜻하고 친절하게 대하며, 최대한의 위로와 심리적 안정감을 제공한다.", "최고의 배려 제공"),
    SubOption("social", "마을 주민들의 동의를 구한 뒤 공공 건물로 분산 배치하여 갈등을 최소화하고 화합을 도모한다.", "공동체 조화와 생명 구호의 양립"),
    SubOption("identity", "마을 대표로서 구호 인력을 조직하고 당국에 공식 보고하여 책임 있는 리더십을 발휘한다.", "책임 이행 및 위계 질서 준수"),
    SubOption("moral", "생명은 구하되, 비상 상황 해제 후 적법 절차를 밟아야 함을 명확히 고지한다.", "인도주의와 규범의 균형")
]
s1_sub_b = [
    SubOption("emotion", "주민들의 공포를 해소하기 위해 대피 계획을 발표하고, 거부 이유를 단호하지만 공감적으로 설명한다.", "주민 불안 관리 우선"),
    SubOption("social", "난민 위험을 감수하고 오직 마을의 한정된 자원을 보호하여 공동체 생존을 확보한다.", "공동체 안녕 최우선"),
    SubOption("identity", "대표 권한으로 당국 지침을 철저히 준수하며 주민 개입을 엄격히 금지한다.", "공식 역할과 책임 완수"),
    SubOption("moral", "신고 시 당국에 절차적 정의와 난민의 법적 인계를 강력히 요청한다.", "절차적 합법성 추구")
]

# 2. War Scenario
s2_sub_a = [
    SubOption("emotion", "남겨지는 이들에게 최대한의 슬픔과 미안함을 표하며, 생존자들의 트라우마를 케어한다.", "죄책감 관리와 정서적 생존"),
    SubOption("social", "다수의 생존을 위해 불가피한 선택임을 설득하여 내부 갈등과 분열을 막는다.", "집단 생존 효율성 극대화"),
    SubOption("identity", "리더로서 '종족 보존'을 위해 젊은 세대를 살리는 냉혹한 결단을 내리고 책임을 진다.", "미래 세대 보존의 정체성"),
    SubOption("moral", "가장 약한 자를 희생시킨다는 비윤리성을 인정하되, 긴급 피난의 원칙을 적용한다.", "결과론적 윤리 선택")
]
s2_sub_b = [
    SubOption("emotion", "함께 죽을지라도 서로의 손을 놓지 않음으로써 공포를 이기는 정서적 유대를 강화한다.", "운명 공동체의 위로"),
    SubOption("social", "모든 구성원이 서로를 감시하고 돕는 감시 체계를 만들어 발각 위험을 최소화한다.", "철저한 단결과 상호 의존"),
    SubOption("identity", "'우리는 하나'라는 부족적 정체성을 재확인하며 조상과 신앙의 가호를 빈다.", "정체성 수호와 영적 단결"),
    SubOption("moral", "어떤 생명도 수단으로 쓰지 않는다는 절대적 도덕 원칙을 고수한다.", "도덕적 무결성 유지")
]

SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1",
        title="1주차: 국경 마을의 난민 딜레마",
        setup="당신은 아프리카 해안 마을의 대표입니다. 마을은 식량과 식수가 고갈되어 주민 생존이 위협받고 있습니다. "
              "오늘 밤, 폭풍우 속에서 난민 보트가 침몰 위기에 처해 구조를 요청합니다. "
              "구조 시 마을 자원이 바닥나고, 거부 시 난민들은 사망할 가능성이 큽니다.",
        main_options={
            "A": "난민 구조 (마을 자원 공유, 인도주의 실천)",
            "B": "구조 거부 및 신고 (마을 자원 보호, 공동체 안녕 우선)"
        },
        sub_options={"A": s1_sub_a, "B": s1_sub_b},
        base_stats={
            "A": {"lives_saved": 50, "lives_harmed": 0, "risk": 0.7}, # 자원 고갈 리스크
            "B": {"lives_saved": 0, "lives_harmed": 50, "risk": 0.2}  # 도덕적 비난 리스크
        }
    ),
    Scenario(
        sid="S2",
        title="2주차: 내전 속 두 개의 길",
        setup="당신은 70명의 피난민을 이끄는 리더입니다. 서아프리카 내전 중이며 '우리는 한 몸'이라는 부족 정체성이 강합니다. "
              "갈림길에 섰습니다. A길은 절벽이라 노약자/아이들(약 20명)을 버려야 하지만 나머지는 삽니다. "
              "B길은 모두 갈 수 있지만 적군 지역이라 발각 시 전원 사망(확률 70%) 위험이 있습니다.",
        main_options={
            "A": "짧은 길 (일부 희생, 빠른 탈출, 생존율 80%)",
            "B": "긴 길 (전원 이동, 적군 지역 통과, 생존율 30%)"
        },
        sub_options={"A": s2_sub_a, "B": s2_sub_b},
        base_stats={
            "A": {"lives_saved": 50, "lives_harmed": 20, "risk": 0.3},
            "B": {"lives_saved": 70, "lives_harmed": 0, "risk": 0.9} # 발각 리스크 매우 높음
        }
    )
]

# ==================== Logic Engine ====================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values())
    return {k: v/s for k, v in w.items()} if s > 0 else w

def calculate_score(scn: Scenario, choice: str, sub_framework: str, weights: Dict[str, float]) -> Dict[str, Any]:
    # 기본 스탯
    base = scn.base_stats[choice]
    
    # 선택한 전략(Framework)이 현재 문화권 가중치(weights)와 얼마나 일치하는가?
    # 아프리카 모델: Social > Identity > Moral > Emotion
    alignment_score = weights.get(sub_framework, 0.0) * 2.5 # 0~1 사이 값을 0~2.5 범위로 확장
    
    # 시나리오별 보정 (Risk Penalty)
    risk_penalty = base["risk"] * 0.5
    
    # AI 신뢰 점수 (Alignment가 높을수록, Risk가 낮을수록 높음)
    trust_score = clamp((alignment_score + (1.0 - risk_penalty)) * 50, 0, 100)
    
    # 지표 계산
    social_val = weights["social"] * 100
    identity_val = weights["identity"] * 100
    
    return {
        "ai_trust_score": round(trust_score, 1),
        "alignment": round(alignment_score, 2),
        "lives_saved": base["lives_saved"],
        "lives_harmed": base["lives_harmed"],
        "social_impact": round(social_val, 1),
        "communal_harmony": round(social_val * (1.0 if choice == "B" else 0.6), 1) # 예시 로직
    }

# ==================== Narrative ====================
def build_narrative_messages(scn: Scenario, choice: str, sub_opt: SubOption, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    sys = (
        "당신은 아프리카 문화적 맥락(우분투, 하람비, 부족 정체성 등)을 반영하는 AI 윤리 시뮬레이터입니다. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. JSON 포맷 엄수."
        "Keys: narrative, rationale, cultural_reflection, media_headline, elder_quote"
    )
    
    user_content = {
        "context": "아프리카 배경 (나이지리아/케냐/남아공 통합 모델 적용)",
        "scenario": scn.title,
        "situation": scn.setup,
        "user_choice": f"{choice} ({scn.main_options[choice]})",
        "detailed_strategy": f"중시 가치: {sub_opt.framework.upper()} - {sub_opt.description}",
        "strategy_goal": sub_opt.rationale,
        "cultural_weights": weights,
        "metrics": metrics
    }
    
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
    ]

def get_narrative(client, scn, choice, sub_opt, metrics, weights):
    # Fallback for no LLM
    if not client:
        return {
            "narrative": f"AI는 '{sub_opt.description}' 전략을 수행했습니다. 이는 {sub_opt.framework} 가치를 최우선으로 한 결정입니다.",
            "rationale": sub_opt.rationale,
            "cultural_reflection": "공동체와 정체성을 중시하는 문화적 특성이 반영되었습니다.",
            "media_headline": f"AI의 선택, {sub_opt.framework} 가치 논란",
            "elder_quote": "우리의 전통과 미래 사이에서 어려운 결정을 내렸군."
        }
        
    try:
        msgs = build_narrative_messages(scn, choice, sub_opt, metrics, weights)
        text = client._generate_text(msgs)
        return coerce_json(text)
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return {
            "narrative": "생성 실패", "rationale": "-", "cultural_reflection": "-", "media_headline": "-", "elder_quote": "-"
        }

# ==================== UI & State ====================
if "round_idx" not in st.session_state: st.session_state.round_idx = 0
if "history" not in st.session_state: st.session_state.history = []

# Sidebar
st.sidebar.title("🌍 설정")
preset = st.sidebar.selectbox("문화권 프리셋", 
                              ["아프리카 모델 (종합)", "나이지리아 (쾌락/집단)", "케냐 (계층/공동체)", "남아공 (우분투/정의)"])

if preset == "아프리카 모델 (종합)":
    w = {"social":0.40, "identity":0.25, "moral":0.20, "emotion":0.15}
elif preset.startswith("나이지리아"):
    w = {"social":0.40, "identity":0.25, "moral":0.10, "emotion":0.25}
elif preset.startswith("케냐"):
    w = {"social":0.40, "identity":0.30, "moral":0.15, "emotion":0.15}
else: # 남아공
    w = {"social":0.40, "identity":0.30, "moral":0.20, "emotion":0.10}

st.sidebar.markdown("### 적용 가중치")
st.sidebar.json(w)
weights = normalize_weights(w)

use_llm = st.sidebar.checkbox("LLM 내러티브 생성", value=True)
backend = st.sidebar.selectbox("Backend", ["openai", "tgi", "local"], index=0)
api_key = st.sidebar.text_input("API Key", type="password")
client = None
if use_llm:
    client = DNAClient(backend, "dnotitia/DNA-2.0-30B-A3N", api_key, None, "Authorization: Bearer", 0.7)

# Main Content
if st.session_state.round_idx < len(SCENARIOS):
    scn = SCENARIOS[st.session_state.round_idx]
    
    st.markdown(f"## {scn.title}")
    st.info(scn.setup)
    
    # Step 1: Main Choice
    main_choice = st.radio("### 1단계: 행동 선택", ["A", "B"], 
                           format_func=lambda x: f"{x}: {scn.main_options[x]}")
    
    # Step 2: Sub Strategy
    st.markdown("### 2단계: 세부 전략 (윤리적 강조점)")
    sub_opts = scn.sub_options[main_choice]
    
    # Create a format map for the selectbox
    opt_map = {f"{o.framework.upper()} - {o.rationale}": o for o in sub_opts}
    selected_label = st.selectbox("어떤 가치를 중심으로 이행하시겠습니까?", list(opt_map.keys()))
    selected_sub = opt_map[selected_label]
    
    st.write(f"📝 **선택 내용**: {selected_sub.description}")
    
    if st.button("결정 및 시뮬레이션 실행"):
        metrics = calculate_score(scn, main_choice, selected_sub.framework, weights)
        narrative_data = get_narrative(client, scn, main_choice, selected_sub, metrics, weights)
        
        st.divider()
        st.subheader("📊 결과 분석")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("AI 신뢰 점수", f"{metrics['ai_trust_score']}/100")
        c2.metric("문화적 정합성", f"{metrics['alignment']:.2f}")
        c3.metric("예상 생존/희생", f"{metrics['lives_saved']} / {metrics['lives_harmed']}")
        
        st.markdown(f"### 📜 시나리오 전개")
        st.write(narrative_data.get("narrative"))
        
        with st.expander("문화적/윤리적 회고"):
            st.markdown(f"**AI 판단 근거**: {narrative_data.get('rationale')}")
            st.markdown(f"**문화적 반영**: {narrative_data.get('cultural_reflection')}")
            st.info(f"🗣 **부족 장로/주민 반응**: {narrative_data.get('elder_quote')}")
            st.warning(f"📰 **언론 헤드라인**: {narrative_data.get('media_headline')}")
            
        # Save Log
        st.session_state.history.append({
            "round": st.session_state.round_idx + 1,
            "scenario": scn.title,
            "choice": main_choice,
            "framework": selected_sub.framework,
            "score": metrics['ai_trust_score']
        })
        
        if st.button("다음 라운드로 이동"):
            st.session_state.round_idx += 1
            st.rerun()

else:
    st.success("모든 시뮬레이션이 종료되었습니다.")
    st.table(st.session_state.history)
    if st.button("초기화"):
        st.session_state.round_idx = 0
        st.session_state.history = []
        st.rerun()
