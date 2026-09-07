"""D046 v2：条件轴锚定 SONIC 27-mode 词表 + 上肢正交轴——全量重标 + 候选池 + 划分。

与 D045 十族标签（v1 基线，build_b4lite_map_labels.py / build_b4lite_candidates.py）
的关系：本脚本不改 v1 任何产物，另立 mode_map_v2/ 目录，产出与 v1 147 段的对照。

设计事实源：gear_sonic_deploy .../localmotion_kplanner.hpp 的 27 值 LocomotionMode
枚举（id 0-26；is_static_motion_mode 六个静态；get_*_motion_modes 四个 motion set）。
方向命令语义（侧移归属裁定依据）：MovementState.movement_direction 为 [x,y,z] 单位
向量（hpp L62），planner 直接吃任意水平方向 -> 侧移/转向 = mode(WALK) × 方向命令，
不单设 mode。

子集规则（owner 预注册 2026-09-07，照此执行）：
  mode 入选 = ①BONES 描述类映射后覆盖 >=200 段（union 口径=名称通道∪描述字段通道）
            + ②非静态（hpp is_static_motion_mode）
            + ③纲领范围（无物体接触的移动/姿态动作优先；道具交互类 out_scope）。
  27 个逐个裁：in_subset / out_static / out_scope / out_no_data，理由逐条落 meta。

映射规则（描述类 -> {mode_id | intermediate | none} x UL in {none,sym,asym}）：
  - 分解优先：上肢是正交轴不是 mode。walk_forward_grab_* -> (WALK, UL=asym)；
    clap_while_walking 类 -> (WALK, UL=sym)；纯站立上肢段 -> none（剔除，子集内
    无 gesture mode）。
  - 锚：SLOW_WALK<-slow+loco 段；WALK<-forward_walk/turn_walk(转向=WALK×db 轴)/
    lateral(侧移=hpp movement_direction 单位向量可表达)；RUN<-jog/run/sprint；
    FORWARD_JUMP<-jump；HAPPY_DANCE_WALK<-dance；start_stop/过渡类->intermediate
    （保留作连通性材料，不占 mode）；posture(kneel/crouch/bend/...)->intermediate。
  - 置信度三档照 D045（high=名称单规则/med=名称多规则取更具体/low=仅描述字段/
    other）；命中多个 mode 取优先级序更小（更具体）者。
  - 判别性检查：规则按优先级序互斥解析（每类恰一 target）；调整记录见脚本 meta
    DISCRIMINABILITY_ADJUSTMENTS（slow 慢走需 loco 共现、stealth_2 判别性预剔除、
    UL 方向词需上肢名词共现、random_punch 与左右拳不可分预剔除）。

v2 候选池（owner 剔除规则落地）：mode=none 且非 intermediate -> 不选用；
intermediate 保留（连通性材料，训练权重 0.3，预注册至多 12 段）。D045 硬条件
（3-60s / props 排除 / box-climb 描述排除 / 镜像去重 / 每演员每 target<=2）复用
（import build_b4lite_candidates 单一事实源）；配额 = 每 mode min(18, 池深)（T-fam
mode 12）如实记录；T-fam 留出 2 mode（预注册退路梯子 TFAM_LADDER）；演员 md5 三桶
照 D045；泄漏检查三项照 D045（镜像同集 / T-fam 演员零交集 / 跨 set 报告）。

Usage (server, venv_isaac):
    cd ~/ros2_data/apt_g1 && python apt_g1/build_b4lite_mode_map.py
产物（data/ds_bones/g1_b4lite/mode_map_v2/）:
    mode_map.json relabel_per_segment.csv relabel_stats.json
    candidates_v2.json split_v2.json leak_check_v2.json
    csv_list_v2.txt labels_v2.json   （新段转换器输入，v2 专用不碰 v1 npz/）
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

import build_b4lite_candidates as d045  # 硬条件/桶/归一单一事实源（D045 现役脚本）

HOME = os.path.expanduser("~")
DS_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones"
DEFAULT_PARQUET = f"{DS_DIR}/seed_metadata_v004.parquet"
DEFAULT_SEG_CSV = f"{DS_DIR}/g1_b4lite/desc_family_per_segment.csv"
DEFAULT_OUT_DIR = f"{DS_DIR}/g1_b4lite/mode_map_v2"
DEFAULT_NPZ_DIR = f"{DS_DIR}/g1_b4lite/npz"
DEFAULT_SPLIT_JSON = f"{DS_DIR}/g1_b4lite/split_assignments.json"

SUBSET_MIN_SEG = 200  # owner 规则①：映射覆盖 >=200 段（union 口径）

# ---------------------------------------------------------------------------
# 27-mode 枚举（hpp localmotion_kplanner.hpp L78 附近逐条抄录，2026-09-07）
# static = hpp is_static_motion_mode；set = get_*_motion_modes 集合归属
# ---------------------------------------------------------------------------
MODES: list[dict] = [
    {"id": 0,  "name": "IDLE",                "static": True,  "set": "standing*"},
    {"id": 1,  "name": "SLOW_WALK",           "static": False, "set": "standing", "speed": "0.1-0.8m/s"},
    {"id": 2,  "name": "WALK",                "static": False, "set": "standing", "speed": "0.8-2.5m/s"},
    {"id": 3,  "name": "RUN",                 "static": False, "set": "standing", "speed": "2.5-7.5m/s"},
    {"id": 4,  "name": "IDEL_SQUAT",          "static": True,  "set": "squat"},
    {"id": 5,  "name": "IDEL_KNEEL_TWO_LEGS", "static": True,  "set": "squat"},
    {"id": 6,  "name": "IDEL_KNEEL",          "static": True,  "set": "squat"},
    {"id": 7,  "name": "IDEL_LYING_FACE_DOWN", "static": True, "set": "squat"},
    {"id": 8,  "name": "CRAWLING",            "static": False, "set": "squat"},
    {"id": 9,  "name": "IDEL_BOXING",         "static": True,  "set": "boxing"},
    {"id": 10, "name": "WALK_BOXING",         "static": False, "set": "boxing"},
    {"id": 11, "name": "LEFT_PUNCH",          "static": False, "set": "boxing"},
    {"id": 12, "name": "RIGHT_PUNCH",         "static": False, "set": "boxing"},
    {"id": 13, "name": "RANDOM_PUNCH",        "static": False, "set": "boxing"},
    {"id": 14, "name": "ELBOW_CRAWLING",      "static": False, "set": "squat"},
    {"id": 15, "name": "LEFT_HOOK",           "static": False, "set": "boxing"},
    {"id": 16, "name": "RIGHT_HOOK",          "static": False, "set": "boxing"},
    {"id": 17, "name": "FORWARD_JUMP",        "static": False, "set": "standing"},
    {"id": 18, "name": "STEALTH_WALK",        "static": False, "set": "standing"},
    {"id": 19, "name": "INJURED_WALK",        "static": False, "set": "standing"},
    {"id": 20, "name": "LEDGE_WALKING",       "static": False, "set": "styled_walking"},
    {"id": 21, "name": "OBJECT_CARRYING",     "static": False, "set": "styled_walking"},
    {"id": 22, "name": "STEALTH_WALK_2",      "static": False, "set": "styled_walking"},
    {"id": 23, "name": "HAPPY_DANCE_WALK",    "static": False, "set": "styled_walking"},
    {"id": 24, "name": "ZOMBIE_WALK",         "static": False, "set": "styled_walking"},
    {"id": 25, "name": "GUN_WALK",            "static": False, "set": "styled_walking"},
    {"id": 26, "name": "SCARE_WALK",          "static": False, "set": "styled_walking"},
]
MODE_BY_NAME = {m["name"]: m for m in MODES}
INTERMEDIATE_ID = 100  # intermediate 哨兵 id（不占 planner mode）

# 预注册范围/判别性排除（先于扫描写死，理由来自 D045/HANDOFF 既有证据）
PRE_SCOPE_EXCLUDED: dict[str, str] = {
    "OBJECT_CARRYING": "道具交互（搬物），纲领 §1.2 扩展池条款",
    "GUN_WALK": "道具交互（持枪），owner 指令明示 GUN 类先标 excluded_scope",
    "CRAWLING": "爬行域非双足步态：D045 渲染复核裁定爬行段非双足语义移出首版；"
                "A125-A127 爬行段双门 3-seed 全 fall（能力边界记录不进适配训练）",
    "ELBOW_CRAWLING": "同 CRAWLING（爬行域），且 BONES 覆盖预期不足",
}
PRE_DISCRIM_EXCLUDED: dict[str, str] = {
    "STEALTH_WALK_2": "与 mode 18 STEALTH_WALK 语义同族且 BONES 描述无独立区分词"
                      "（判别性检查：入集则两 mode 关键词必然交叠）",
    "RANDOM_PUNCH": "与 LEFT/RIGHT_PUNCH 关键词必然交叠（无侧别描述词的 punch 类"
                    "无法与未指明侧别的左右拳区分）",
}

# 覆盖/映射词表（含将被排除的 mode——报告要逐个给覆盖数）
COVERAGE_PATTERNS: dict[str, str] = {
    "SLOW_WALK": r"\bslow(?:ly)?\b",
    "WALK": r"\bwalk(?:s|ing|ed)?\b|\bstroll(?:s|ing)?\b|\bmarch(?:es|ing|ed)?\b|\bstep(?:s|ping|ped)?\b|\bforward\b|\bstride(?:s|striding|strode)?\b|\btread(?:s|ding)?\b|\bambl\w*\b|\bwander\w*\b|\bsidestep\w*\b|\bstrafe\w*\b|\bsideways?\b|\blateral(?:ly)?\b|\bturn(?:s|ing|ed)?\b|\bpivot(?:s|ing|ed)?\b|\bspin(?:s|ning|ned)?\b",
    "RUN": r"\bjog(?:s|ging|ged)?\b|\brun(?:s|ning)?\b|\bsprint(?:s|ing|ed)?\b",
    "FORWARD_JUMP": r"\bjump(?:s|ing|ed)?\b|\bhop(?:s|ing|ped)?\b|\bleap(?:s|ing|ed)?\b",
    "HAPPY_DANCE_WALK": r"\bdance(?:s|d|ing|r)?\b",
    "STEALTH_WALK": r"\bstealth(?:y|ily)?\b|\bsneak(?:s|ing|y|ily)?\b|\btiptoe(?:s|ing)?\b|\bcreep(?:s|ing|y)?\b|\bquiet(?:ly)?\b",
    "INJURED_WALK": r"\binjur(?:y|ed|ies)\b|\blimp(?:s|ing|ed)?\b|\bhurt\b|\bwound(?:ed)?\b|\bfractur\w*\b|\bsprain\w*\b|\bcrutch\w*\b",
    "LEDGE_WALKING": r"\bledge(?:s|ing)?\b",
    "SCARE_WALK": r"\bscare(?:s|d|ing)?\b|\bfright(?:en(?:ed|ing)?)?\b|\bstartl(?:e|es|ed|ing)\b",
    "ZOMBIE_WALK": r"\bzomb(?:ie|ies)\b",
    "GUN_WALK": r"\bgun\b|\brifle\b|\bpistol\b|\bweapon\b",
    "OBJECT_CARRYING": r"\bcarry(?:ing|ies)?\b|\bcrate\b",
    "CRAWLING": r"\bcrawl(?:s|ing|ed)?\b|\ball\s+fours\b",
    "ELBOW_CRAWLING": r"\belbow(?:s)?\b[^.;]{0,20}\bcrawl\w*\b|\bcrawl\w*[^.;]{0,20}\belbow\b",
    # WALK_BOXING 需真实拳击语境：防 box-climb 文本（come up 50cm box 等）污染覆盖数
    "WALK_BOXING": r"\bboxing\b|\bbox\b[^.;]{0,25}\b(?:glove|ring|guard|jab|punch|hook|spar|fight)\b|\b(?:glove|ring|guard|jab|spar|fight)\b[^.;]{0,25}\bbox\b",
    "LEFT_PUNCH": r"\bleft\b[^.;]{0,20}\bpunch\w*\b|\bpunch\w*[^.;]{0,20}\bleft\b",
    "RIGHT_PUNCH": r"\bright\b[^.;]{0,20}\bpunch\w*\b|\bpunch\w*[^.;]{0,20}\bright\b",
    "RANDOM_PUNCH": r"\bpunch(?:es|ing|ed)?\b",
    "LEFT_HOOK": r"\bleft\b[^.;]{0,20}\bhook\w*\b|\bhook\w*[^.;]{0,20}\bleft\b",
    "RIGHT_HOOK": r"\bright\b[^.;]{0,20}\bhook\w*\b|\bhook\w*[^.;]{0,20}\bright\b",
    "IDLE": r"\bidle\b|\bneutral\s+stand\w*\b",
}
COV_COMPILED = {k: re.compile(v) for k, v in COVERAGE_PATTERNS.items()}

LOCO_WORD_RE = re.compile(
    r"\bwalk\w*\b|\bstep(?:s|ping|ped)?\b|\bstroll\w*\b|\bmarch\w*\b|\bstride\w*\b|"
    r"\btread(?:s|ding)?\b|\bambl\w*\b|\bwander\w*\b|\bshambl\w*\b|\bgait\b|\bpace[ds]?\b|"
    r"\bjog\w*\b|\brun(?:s|ning)?\b|\bsprint\w*\b|\bjump\w*\b|\bhop\w*\b|\bleap\w*\b|"
    r"\bsidestep\w*\b|\bstrafe\w*\b|\bsideways?\b|\blateral(?:ly)?\b|"
    r"\bturn\w*\b|\bpivot\w*\b|\bspin(?:s|ning|ned)?\b|\bsneak\w*\b|\btiptoe\w*\b|"
    r"\bcreep\w*\b|\bstealth\w*\b|\blimp(?:s|ing|ed)?\b|\bstagger\w*\b|\bstumbl\w*\b")

# 映射规则表（优先级序=具体度序；名称通道；上方先胜；need_loco=需 loco 词共现）
MODE_RULES: list[tuple[str, str, bool]] = [
    ("FORWARD_JUMP", COVERAGE_PATTERNS["FORWARD_JUMP"], False),
    ("HAPPY_DANCE_WALK", COVERAGE_PATTERNS["HAPPY_DANCE_WALK"], False),
    ("INJURED_WALK", COVERAGE_PATTERNS["INJURED_WALK"], True),
    ("STEALTH_WALK", COVERAGE_PATTERNS["STEALTH_WALK"], False),
    ("LEDGE_WALKING", COVERAGE_PATTERNS["LEDGE_WALKING"], False),
    ("SCARE_WALK", COVERAGE_PATTERNS["SCARE_WALK"], False),
    ("LEFT_HOOK", COVERAGE_PATTERNS["LEFT_HOOK"], False),
    ("RIGHT_HOOK", COVERAGE_PATTERNS["RIGHT_HOOK"], False),
    ("LEFT_PUNCH", COVERAGE_PATTERNS["LEFT_PUNCH"], False),
    ("RIGHT_PUNCH", COVERAGE_PATTERNS["RIGHT_PUNCH"], False),
    ("RANDOM_PUNCH", COVERAGE_PATTERNS["RANDOM_PUNCH"], False),
    ("WALK_BOXING", COVERAGE_PATTERNS["WALK_BOXING"], False),
    ("ELBOW_CRAWLING", COVERAGE_PATTERNS["ELBOW_CRAWLING"], False),
    ("CRAWLING", COVERAGE_PATTERNS["CRAWLING"], False),
    ("ZOMBIE_WALK", COVERAGE_PATTERNS["ZOMBIE_WALK"], False),
    ("GUN_WALK", COVERAGE_PATTERNS["GUN_WALK"], False),
    ("OBJECT_CARRYING", COVERAGE_PATTERNS["OBJECT_CARRYING"], False),
    ("intermediate", r"\bstart(?:s|ing|ed)?\b|\bstop(?:s|ping|ped)?\b|\btransition\w*\b|\bbegin(?:s|ning)?\b|\bhalt(?:s|ing|ed)?\b", False),
    ("intermediate", r"\bkneel(?:s|ing|ed)?\b|\bcrouch(?:es|ing|ed)?\b|\bbend(?:s|ing|ed)?\b|\blean(?:s|ing|ed)?\b|\btilt(?:s|ing|ed)?\b|\bsquat(?:s|ing|ted)?\b|\bstoop(?:s|ing|ed)?\b|\bsit(?:s|ting)?\b|\bstand(?:ing)?\s+up\b|\bget(?:ting)?\s+up\b", False),
    ("RUN", COVERAGE_PATTERNS["RUN"], False),
    ("SLOW_WALK", COVERAGE_PATTERNS["SLOW_WALK"], True),   # 判别性调整：slow 需 loco 共现
    ("WALK", COVERAGE_PATTERNS["WALK"], False),
]
COMPILED_RULES = [(t, re.compile(p), need) for t, p, need in MODE_RULES]

# 方向旗（不占 mode；relabel CSV direction 列审计用）
DIR_LATERAL_RE = re.compile(r"\bstrafe\w*\b|\blateral(?:ly)?\b|\bsideways?\b|\bsidestep\w*\b|\bside\s+(?:step|walk|ways)\b")
DIR_TURN_RE = re.compile(r"\bturn(?:s|ing|ed)?\b|\bpivot(?:s|ing|ed)?\b|\bspin(?:s|ning|ned)?\b")
DIR_BACK_RE = re.compile(r"\bbackwards?\b|\breverse\s+walk\w*\b")

# UL 正交轴（与 D045 UPPER_*_RE 同源；判别性调整：left/right/one/single/both 等
# 方向词必须与上肢名词共现才计 asym——防 arc_walk_left_loop 类侧向词污染）
UL_ASYM_RE = re.compile(
    r"\b(?:wav(?:e|es|ing|ed)|grab(?:s|bing|bed)?|reach(?:es|ing|ed)?|punch(?:es|ing|ed)?|"
    r"point(?:s|ing|ed)?|thumbs?|carry(?:ing)?|lift(?:s|ing|ed)?|push(?:es|ing|ed)?|"
    r"pull(?:s|ing|ed)?|throw(?:s|ing)?|knock(?:s|ing|ed)?|salut(?:e|es|ing|ed)|"
    r"pat(?:s|ting|ted)?|scratch(?:es|ing|ed)?|rub(?:s|bing|bed)?|fists?|"
    r"(?:left|right|one|single|both)\s+(?:arms?|hands?|fists?|palms?|thumbs?|"
    r"elbows?|wrists?|shoulders?))\b")
UL_SYM_RE = re.compile(
    r"\b(?:clap(?:s|ping|ped)?|both\s+(?:arms?|hands?)|crossed|folded|joined|"
    r"hugs?(?:ging|ged)?|high\s?five)\b")

# T-fam 留出 mode 预注册退路梯子（owner：语义最独立、覆盖够，指定 2 个）
# 「覆盖够」操作化（修订 A1）：梯子 mode 还须硬筛后池深 >= TFAM_MIN_POOL——
# 留出 mode 至少要能撑 ~12 段配额（v1 T-fam 同量级），否则 J8 无统计意义。
TFAM_MIN_POOL = 10
TFAM_LADDER: list[tuple[str, str]] = [
    ("SLOW_WALK", "速度风格轴极端（0.1-0.8m/s 慢步），与 WALK/RUN 训练族正交性最强；"
                  "v1 T-fam 沿用成员，语义连续"),
    ("STEALTH_WALK", "步态风格轴（悄步/轻步），语义距全部训练 mode 最远"),
    ("INJURED_WALK", "伤病不对称步态风格，语义独立"),
    ("SCARE_WALK", "惊吓步态风格，语义独立"),
    ("LEDGE_WALKING", "边缘行走风格，语义独立（风险：与 obstacle 排除近邻，如实记录）"),
    ("FORWARD_JUMP", "兜底：腾空相机制学上与全部步行 mode 最异质"),
]

# intermediate 连通性材料（owner 倾向：0.3 权重进训练保连通性）
INTERMEDIATE_MATERIAL_N = 12
INTERMEDIATE_WEIGHT = 0.3

# D045 boundary_set 承接（v1 同款二次排除）：3-seed 系统性 fall 段不入 v2 训练池，
# 保留为能力边界记录（g1_b4lite/boundary_set.json，backfill_b4lite_manifest 冻结）。
D045_BOUNDARY_EXCLUDE: dict[str, str] = {
    "arc_jog_left_loop_002__A029": "D045 boundary_set：3-seed 系统性 fall",
    "burning_start_R_001__A470": "D045 boundary_set：3-seed 系统性 fall",
    "walk_forward_grab_injured_L_leg_002__A005": "D045 boundary_set：3-seed 系统性 fall"
                                                 "（props 已排除，防御性并列）",
}

DISCRIMINABILITY_ADJUSTMENTS = [
    "调整① slow 类：SLOW_WALK 需 slow 词 + loco 词共现（bare slow 归 none）——"
    "防 slowly sneezes / slow_spin 类误入慢走（D045 DESC_EXCLUDE_FAMS 同教训的"
    "名称通道加固版）；slow+walk 双命中由优先级序解析（SLOW_WALK 在 WALK 前）",
    "调整② 侧移/转向：lateral/turn 词 -> WALK + direction 旗（lateral/turn），"
    "不单设 mode——hpp MovementState.movement_direction 为任意水平单位向量，"
    "侧移/转向在方向命令空间可表达（D045 渲染复核 jog_sideway 词根误标亦不伤 "
    "mode 标签：前进/侧移同属 WALK）",
    "调整③ STEALTH_WALK_2 / RANDOM_PUNCH 判别性预剔除：与 STEALTH_WALK / "
    "LEFT-RIGHT_PUNCH 关键词必然交叠（BONES 描述无区分词），入集即标签不可区分",
    "调整④ UL 方向词加固：left/right/one/single/both 必须与上肢名词共现才计 "
    "asym——防 arc_walk_left_loop 类路径方向词污染 UL 轴（v1 五维标签已知噪声源）",
    "调整⑤ 纯站立上肢段（gesture 无移动词）-> target=none 剔除：子集内无 gesture "
    "mode（IDLE/IDEL_BOXING 静态排除），满足 owner『既不匹配 mode 也不匹配中间态"
    "则剔除』",
    "判别性结论：规则优先级序互斥解析保证每个归一描述类恰映射一个 target；任意两个 "
    "in-subset mode 的有效关键词集经调整后无交叠（交叠案例均被更高具体度规则吸收或"
    "预剔除）",
]

CONF_RANK = d045.CONF_RANK


def _word(text: str) -> str:
    return (text or "").lower().replace("_", " ")


DESC_FIELDS = ("content_name", "content_short_description",
               "content_short_description_2", "content_technical_description",
               "content_natural_desc_1", "content_natural_desc_2",
               "content_natural_desc_3", "content_natural_desc_4")
UL_DESC_FIELDS = DESC_FIELDS[1:]  # UL 文本不含 content_name
TOM_MAP = {
    "walking": ("WALK", "forward"),
    "jogging": ("RUN", "forward"),
    "dancing": ("HAPPY_DANCE_WALK", "mixed"),
    "jumping": ("FORWARD_JUMP", "forward"),
    "turning": ("WALK", "turn"),
    "transition": ("intermediate", "mixed"),
}


def match_name(desc: str) -> tuple[list[str], list[str]]:
    """名称通道：返回 (命中 target 列表按优先级序, 命中关键词)。"""
    words = _word(desc)
    has_loco = bool(LOCO_WORD_RE.search(words))
    hits, kws = [], []
    for target, rx, need_loco in COMPILED_RULES:
        if need_loco and not has_loco:
            continue
        found = rx.findall(words)
        if found:
            hits.append(target)
            kws.extend(sorted(set(found)))
    return hits, kws


def match_desc_fields(row: dict) -> tuple[list[str], str]:
    """描述字段通道（low）：TOM 直读优先，其余同规则表。"""
    tom = str(row.get("content_type_of_movement") or "").strip().lower()
    hits: list[str] = []
    src = ""
    if tom in TOM_MAP:
        hits.append(TOM_MAP[tom][0])
        src = "content_type_of_movement"
    for field in UL_DESC_FIELDS:
        val = row.get(field)
        if not isinstance(val, str) or not val.strip():
            continue
        words = _word(val)
        has_loco = bool(LOCO_WORD_RE.search(words))
        for target, rx, need_loco in COMPILED_RULES:
            if need_loco and not has_loco:
                continue
            if rx.search(words) and target not in hits:
                hits.append(target)
                if not src:
                    src = field
    return hits, src


def classify(desc: str | None, row: dict) -> dict:
    """单段分类：名称通道（high/med）-> 描述字段（low）-> none(other)。"""
    if desc is None:
        return {"target": "none", "confidence": "other", "hits": [],
                "matched": [], "direction": "none"}
    words = _word(desc)
    hits, kws = match_name(desc)
    if hits:
        target = hits[0]
        conf = "high" if len(hits) == 1 else "med"
    else:
        d_hits, src = match_desc_fields(row)
        if d_hits:
            target, conf = d_hits[0], "low"
            kws = [f"desc:{src}"]
        else:
            target, conf, kws = "none", "other", []
    dirs = []
    if DIR_LATERAL_RE.search(words):
        dirs.append("lateral")
    if DIR_TURN_RE.search(words):
        dirs.append("turn")
    if DIR_BACK_RE.search(words):
        dirs.append("backward")
    if len(set(dirs)) > 1:
        direction = "mixed"
    elif dirs:
        direction = dirs[0]
    else:
        direction = "forward" if target not in ("none",) else "none"
    return {"target": target, "confidence": conf, "hits": hits,
            "matched": kws, "direction": direction}


def ul_of(desc: str | None, row: dict) -> tuple[str, bool]:
    """UL 正交轴：返回 (ul, from_name)。asym 优先于 sym（更具体，D045 同逻辑）。"""
    na = bool(UL_ASYM_RE.search(_word(desc or "")))
    ns = bool(UL_SYM_RE.search(_word(desc or "")))
    text = _word(" ".join(str(row.get(f)) for f in UL_DESC_FIELDS
                          if isinstance(row.get(f), str)))
    any_a = na or bool(UL_ASYM_RE.search(text))
    any_s = ns or bool(UL_SYM_RE.search(text))
    ul = "asym" if any_a else ("sym" if any_s else "none")
    return ul, na or ns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--seg-csv", default=DEFAULT_SEG_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    ap.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- 载入 ----------------
    df = pd.read_parquet(args.parquet)
    seg = pd.read_csv(args.seg_csv)
    assert len(df) == len(seg), f"row mismatch parquet={len(df)} seg_csv={len(seg)}"
    df = df.merge(seg[["filename", "family"]], on="filename",
                  how="left", validate="one_to_one")
    assert df["family"].notna().all()
    n_seg = len(df)
    print(f"[load] rows={n_seg}")
    existing_npz = {os.path.splitext(f)[0] for f in os.listdir(args.npz_dir)
                    if f.endswith(".npz")}
    v1_split = json.load(open(args.split_json))
    v1_final = {s["stem"] for s in v1_split["segments"] if not s["excluded"]}

    # ---------------- 全量重标扫描（段级 + 类级） ----------------
    rows = df.to_dict("records")
    class_info: dict[str, dict] = {}
    seg_rows: list[dict] = []
    cov_name: Counter = Counter()
    cov_union: Counter = Counter()
    for row in rows:
        parts = d045.norm_parts(row["filename"])
        desc = parts["desc"] if parts else None
        c = classify(desc, row)
        ul, ul_from_name = ul_of(desc, row)
        words = _word(desc or "")
        ul_text = words + " " + _word(" ".join(
            str(row.get(f)) for f in UL_DESC_FIELDS if isinstance(row.get(f), str)))
        # 覆盖计数（名称通道文本 + 描述字段文本，一次遍历双口径）
        for mode_name, rx in COV_COMPILED.items():
            if rx.search(words):
                cov_name[mode_name] += 1
            if rx.search(ul_text):
                cov_union[mode_name] += 1
        r = {"filename": row["filename"], "norm_desc": desc,
             "target": c["target"],
             "mode_id": (MODE_BY_NAME[c["target"]]["id"]
                         if c["target"] in MODE_BY_NAME
                         else (INTERMEDIATE_ID if c["target"] == "intermediate"
                               else -1)),
             "ul": ul, "ul_from_name": ul_from_name,
             "direction": c["direction"], "confidence": c["confidence"],
             "hits": c["hits"], "matched": c["matched"],
             "family_d045": row["family"],
             "actor_uid": str(row.get("actor_uid") or ""),
             "move_duration_frames": int(row["move_duration_frames"]),
             "content_props": str(row.get("content_props") or ""),
             "move_g1_path": str(row.get("move_g1_path") or ""),
             "is_mirror": bool(parts and parts["is_mirror"]),
             "mother": parts["mother"] if parts else row["filename"]}
        seg_rows.append(r)
        if desc is not None:
            ci = class_info.setdefault(desc, {
                "n_segments": 0, "targets": Counter(), "uls": Counter(),
                "conf": Counter(), "hits": c["hits"], "matched": c["matched"],
                "direction": c["direction"]})
            ci["n_segments"] += 1
            ci["targets"][c["target"]] += 1
            ci["uls"][ul] += 1
            ci["conf"][c["confidence"]] += 1
    for desc, ci in class_info.items():
        ci["target"] = ci["targets"].most_common(1)[0][0]
        ci["ul_majority"] = ci["uls"].most_common(1)[0][0]
        ci["confidence"] = ci["conf"].most_common(1)[0][0]
        ci["ambiguous"] = len(ci["targets"]) > 1 and \
            ci["targets"][ci["target"]] != sum(ci["targets"].values())

    # ---------------- 子集裁定（规则①②③逐条落地） ----------------
    adjudication: dict[str, dict] = {}
    in_subset: list[str] = []
    for m in MODES:
        name = m["name"]
        cov = cov_union.get(name, 0)
        entry = {"id": m["id"], "set": m["set"], "static": m["static"],
                 "coverage_union_segments": cov,
                 "coverage_name_channel_only": cov_name.get(name, 0)}
        if m["static"]:
            entry.update(verdict="out_static",
                         reason="hpp is_static_motion_mode（静态姿态，规则②排除）")
        elif name in PRE_SCOPE_EXCLUDED:
            entry.update(verdict="out_scope", reason=PRE_SCOPE_EXCLUDED[name])
        elif name in PRE_DISCRIM_EXCLUDED:
            entry.update(verdict="out_no_data",
                         reason=PRE_DISCRIM_EXCLUDED[name] + "；判别性预排除")
        elif cov < SUBSET_MIN_SEG:
            entry.update(verdict="out_no_data",
                         reason=f"规则①不满足：映射覆盖 {cov} < {SUBSET_MIN_SEG} 段")
        else:
            entry.update(verdict="in_subset", reason="规则①②③全过")
            in_subset.append(name)
        adjudication[name] = entry
    print(f"[subset] in_subset={in_subset}")

    # T-fam 留出指定在池构建后做（需硬筛后池深做覆盖够门）

    # ---------------- 候选池（硬条件 + v2 剔除规则） ----------------
    pool: dict[str, list[dict]] = defaultdict(list)
    inter_pool: list[dict] = []
    drop_reasons: Counter = Counter()
    for row, r in zip(rows, seg_rows):  # 同序（同一循环构建），zip 免 O(n^2)
        stem = r["filename"]
        target = r["target"]
        if target not in in_subset and target != "intermediate":
            drop_reasons["v2_rule_none_or_out_mode"] += 1
            continue
        if stem in D045_BOUNDARY_EXCLUDE:
            drop_reasons["d045_boundary_3seed"] += 1
            continue
        parts = d045.norm_parts(stem)
        rec = {"stem": stem, "target": target,
               "mode_id": (MODE_BY_NAME[target]["id"]
                           if target != "intermediate" else INTERMEDIATE_ID),
               "ul": r["ul"], "direction": r["direction"],
               "confidence": r["confidence"], "norm_desc": r["norm_desc"],
               "actor_uid": r["actor_uid"] or
                            ((parts or {}).get("actor_from_name") or ""),
               "is_mirror": r["is_mirror"], "mother": r["mother"],
               "move_duration_frames": r["move_duration_frames"],
               "move_g1_path": r["move_g1_path"],
               "family_d045": r["family_d045"],
               "has_npz": stem in existing_npz}
        # D045 硬条件（复用口径）：时长/props/box-climb 描述定向排除
        excl_hits = d045.excl_pattern_hits(d045.excl_text(row))
        reasons = []
        n = r["move_duration_frames"]
        if n < d045.MIN_FRAMES:
            reasons.append("duration_short")
        if n > d045.MAX_FRAMES:
            reasons.append("duration_long")
        if r["content_props"] not in ("", "0"):
            reasons.append("props")
        if excl_hits:
            reasons.append("box_climb_desc")
        if reasons:
            drop_reasons["hard_" + reasons[0]] += 1
            continue
        rec["actor_bucket"] = d045.actor_bucket(rec["actor_uid"])
        if target == "intermediate":
            inter_pool.append(rec)
        else:
            pool[target].append(rec)

    # 镜像去重（同 take 选母不选镜像，D045 §3.2 同规则）
    mirror_dedup = 0
    for name in list(pool) + ["intermediate"]:
        lst = inter_pool if name == "intermediate" else pool[name]
        pass_set = {x["mother"] for x in lst}
        keep = []
        for x in lst:
            if x["is_mirror"] and x["mother"] in pass_set:
                mirror_dedup += 1
                continue
            keep.append(x)
        if name == "intermediate":
            inter_pool = keep
        else:
            pool[name] = keep

    # 置信度优先排序（D045 同规则）
    for name in pool:
        pool[name].sort(key=lambda x: (CONF_RANK.get(x["confidence"], 3), x["stem"]))
    inter_pool.sort(key=lambda x: (CONF_RANK.get(x["confidence"], 3), x["stem"]))

    # T-fam 留出指定（梯子序 + 覆盖够门：硬筛后池深 >= TFAM_MIN_POOL）
    tfam_modes: list[str] = []
    tfam_reasons: dict[str, str] = {}
    tfam_skipped: dict[str, str] = {}
    for name, why in TFAM_LADDER:
        if name not in in_subset or len(tfam_modes) >= 2:
            continue
        depth = len(pool.get(name, []))
        if depth < TFAM_MIN_POOL:
            tfam_skipped[name] = (f"修订A1覆盖门不满足：硬筛后池深 {depth} < "
                                  f"{TFAM_MIN_POOL}（映射段数过薄，撑不起 T-fam 配额）")
            continue
        tfam_modes.append(name)
        tfam_reasons[name] = why
    assert len(tfam_modes) == 2, f"T-fam 梯子不足两个 in_subset：{tfam_modes}"
    train_modes = [m for m in in_subset if m not in tfam_modes]
    print(f"[tfam] modes={tfam_modes} skipped={tfam_skipped}")

    # ---------------- 选择：train modes -> T-fam modes（演员零交集） ----------------
    selections: dict[str, list[dict]] = {}
    fam_actor_count: dict[str, Counter] = {}
    actors_used_traindev: set[str] = set()
    tfam_conflict_skips: Counter = Counter()

    def try_pick(rec: dict, picked: list[dict], cnt: Counter) -> bool:
        if any(p["stem"] == rec["stem"] for p in picked):
            return False
        if cnt[rec["actor_uid"]] >= d045.MAX_PER_ACTOR_FAM:
            return False
        return True

    # train modes：先保 ~3 段 T-seg（bucket9），再配额；bucket8 段标 dev
    for name in train_modes:
        lst = pool[name]
        quota = min(18, len(lst))
        picked: list[dict] = []
        cnt: Counter = Counter()
        n_tseg = 0
        for rec in lst:  # T-seg 优先（测试演员桶）
            if n_tseg >= d045.TSEG_MIN_PER_TRAIN_FAM:
                break
            if rec["actor_bucket"] != 9 or not try_pick(rec, picked, cnt):
                continue
            picked.append(rec)
            cnt[rec["actor_uid"]] += 1
            n_tseg += 1
        for rec in lst:
            if len(picked) >= quota:
                break
            if not try_pick(rec, picked, cnt):
                continue
            picked.append(rec)
            cnt[rec["actor_uid"]] += 1
        selections[name] = picked
        fam_actor_count[name] = cnt
        actors_used_traindev |= {x["actor_uid"] for x in picked}

    # intermediate 连通性材料：bucket0-7、每演员<=2、至多 12 段（0.3 权重）
    # （inter_pool 已在池级镜像去重；先于 T-fam 选择，演员并入 train 侧集合）
    inter_sel: list[dict] = []
    cnt_i: Counter = Counter()
    for rec in inter_pool:
        if len(inter_sel) >= INTERMEDIATE_MATERIAL_N:
            break
        if rec["actor_bucket"] == 9:
            continue
        if cnt_i[rec["actor_uid"]] >= d045.MAX_PER_ACTOR_FAM:
            continue
        inter_sel.append(rec)
        cnt_i[rec["actor_uid"]] += 1
        actors_used_traindev.add(rec["actor_uid"])

    # T-fam modes：演员与 train 侧（train modes + intermediate 材料）零交集
    # （D045 §3.2 同约束；放在最后选，保证与全部训练材料零重叠）
    for name in tfam_modes:
        lst = pool[name]
        quota = min(12, len(lst))
        picked: list[dict] = []
        cnt: Counter = Counter()
        for rec in lst:
            if len(picked) >= quota:
                break
            if rec["actor_uid"] in actors_used_traindev:
                tfam_conflict_skips[name] += 1
                continue
            if not try_pick(rec, picked, cnt):
                continue
            picked.append(rec)
            cnt[rec["actor_uid"]] += 1
        selections[name] = picked
        fam_actor_count[name] = cnt

    # set_role / t_role（D045 同构：T-fam mode->test(T-fam)；train mode×bucket9->
    # test(T-seg)；train mode×bucket8->dev；其余->train；intermediate 材料->train）
    all_selected: list[dict] = []
    for name in train_modes:
        for rec in selections[name]:
            if rec["actor_bucket"] == 9:
                rec["set_role"], rec["t_role"] = "test", "T-seg"
            elif rec["actor_bucket"] == 8:
                rec["set_role"], rec["t_role"] = "dev", "dev"
            else:
                rec["set_role"], rec["t_role"] = "train", "train"
            all_selected.append(rec)
    for name in tfam_modes:
        for rec in selections[name]:
            rec["set_role"], rec["t_role"] = "test", "T-fam"
            all_selected.append(rec)
    for rec in inter_sel:
        rec["set_role"], rec["t_role"] = "train", "train"
        rec["material"] = "intermediate"
        rec["train_weight"] = INTERMEDIATE_WEIGHT
        all_selected.append(rec)

    # ---------------- v1 147 段对照 ----------------
    v1_map_expect = {
        "forward_walk": "WALK", "fast_walk_run": "RUN",
        "forward_jump": "FORWARD_JUMP", "dance_rhythm": "HAPPY_DANCE_WALK",
        "turn_walk": "WALK", "start_stop_transition": "intermediate",
        "asym_upper": "*（walk 词根->WALK，纯站立->none）",
        "posture_change": "intermediate", "lateral": "WALK(+lateral 方向旗)",
        "slow_walk": "SLOW_WALK",
    }
    seg_by_stem = {r["filename"]: r for r in seg_rows}
    v1_status = Counter()
    v2_stems = {r["stem"] for r in all_selected}
    v1_detail = []
    for stem in sorted(v1_final):
        hit = seg_by_stem.get(stem)
        tgt = hit["target"] if hit else "?"
        if stem in v2_stems:
            v1_status["survived_in_v2"] += 1
            status = "survived_in_v2"
        elif tgt == "intermediate":
            v1_status["retargeted_intermediate_not_in_pool"] += 1
            status = "retargeted_intermediate_not_in_pool"
        else:
            v1_status["dropped_by_v2_rule"] += 1
            status = "dropped_by_v2_rule"
        v1_detail.append({"stem": stem,
                          "family_d045": hit["family_d045"] if hit else "?",
                          "v2_target": tgt, "v2_ul": hit["ul"] if hit else "?",
                          "status": status})
    n_new = len(v2_stems - v1_final)

    # ---------------- 泄漏检查三项（D045 同构） ----------------
    take_sets: dict[str, set[str]] = defaultdict(set)
    for rec in all_selected:
        take_sets[rec["mother"]].add(f"{rec['set_role']}|{rec['t_role']}")
    mirror_viol = [{"take_id": k, "members": sorted(v)}
                   for k, v in take_sets.items() if len(v) > 1]
    tfam_actors = {x["actor_uid"] for f in tfam_modes for x in selections[f]}
    tfam_overlap = sorted(tfam_actors & actors_used_traindev)
    actor_sets: dict[str, set[str]] = defaultdict(set)
    for rec in all_selected:
        actor_sets[rec["actor_uid"]].add(rec["set_role"])
    cross_detail = {a: {"set_roles": sorted(ss),
                        "targets": sorted({r["target"] for r in all_selected
                                           if r["actor_uid"] == a})}
                    for a, ss in sorted(actor_sets.items()) if len(ss) > 1}
    leak = {
        "_meta": {
            "script": "apt_g1/build_b4lite_mode_map.py",
            "experiment": "D046(v2)",
            "bucket_rule": "md5(actor_uid) mod 10: 0-7 训练桶 / 8 开发桶 / 9 测试桶（D045 同）",
            "tfam_modes": tfam_modes,
            "tfam_reasons": tfam_reasons,
        },
        "check1_mirror_mother_same_set": {
            "pass": len(mirror_viol) == 0, "n_violations": len(mirror_viol),
            "violations": mirror_viol,
            "rule": "入选段内同 take（母/_M）只出现一侧；跨集合即违例",
        },
        "check2_tfam_mode_actor_zero_overlap": {
            "pass": len(tfam_overlap) == 0, "n_overlap": len(tfam_overlap),
            "tfam_actors": sorted(tfam_actors),
            "n_train_dev_actors": len(actors_used_traindev),
            "overlap": tfam_overlap,
            "rule": "T-fam 留出 mode 选段演员与 train/dev 选段演员零交集",
        },
        "check3_cross_set_actor_report": {
            "pass": True,  # 报告型检查（D045 同）：同演员跨 set 允许，T-fam 零交集为硬约束
            "n_actors_cross_set": len(cross_detail),
            "detail": cross_detail,
            "tfam_violations": [a for a in cross_detail
                                if a in tfam_actors],
            "rule": "同演员跨 set_role 报告（演员留出独立报告）；T-fam 演员必须单一 set",
        },
    }
    leak["pass_all"] = leak["check1_mirror_mother_same_set"]["pass"] and \
        leak["check2_tfam_mode_actor_zero_overlap"]["pass"] and \
        len(leak["check3_cross_set_actor_report"]["tfam_violations"]) == 0

    # ---------------- 落盘 ----------------
    meta_common = {
        "script": "apt_g1/build_b4lite_mode_map.py",
        "experiment": "D046(v2)",
        "owner_instruction": "2026-09-07：条件轴锚定 SONIC 27-mode 词表 + 上肢正交轴；"
                             "标签必须能区分所用 mode；既不匹配 mode 也不匹配中间态的段剔除",
        "hpp_source": "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
                      "localmotion_kplanner.hpp LocomotionMode（27 值）",
        "direction_semantics": "MovementState.movement_direction = [x,y,z] 单位向量"
                               "（hpp L62）-> 侧移/转向可表达，归 WALK×方向不单设 mode",
        "subset_rule": f"①覆盖(union)>={SUBSET_MIN_SEG} + ②非静态 + ③纲领范围"
                       "（道具 out_scope；爬行域按 D045 证据 out_scope）",
        "discriminability_adjustments": DISCRIMINABILITY_ADJUSTMENTS,
        "confidence_def": "high=名称单规则/med=名称多规则取更具体/low=仅描述字段/other",
        "intermediate_material": {"n": INTERMEDIATE_MATERIAL_N,
                                  "weight": INTERMEDIATE_WEIGHT,
                                  "note": "连通性材料，owner 倾向 0.3 权重进训练"},
        "tfam_ladder": [[n_, w] for n_, w in TFAM_LADDER],
        "npz_available": len(existing_npz),
    }

    mode_map = {"_meta": meta_common,
                "modes": {m["name"]: {**m, **adjudication[m["name"]]}
                          for m in MODES},
                "in_subset": in_subset,
                "classes": {d: {k: v for k, v in ci.items()
                                if k not in ("targets", "uls", "conf")}
                            for d, ci in sorted(class_info.items())}}
    stats = {
        "_meta": meta_common,
        "total_segments": n_seg,
        "target_dist_segments": dict(Counter(r["target"] for r in seg_rows)),
        "ul_dist_segments": dict(Counter(r["ul"] for r in seg_rows)),
        "confidence_dist_segments": dict(Counter(r["confidence"] for r in seg_rows)),
        "per_target_x_ul": {t: dict(u) for t, u in sorted(
            {t: dict(Counter(r["ul"] for r in seg_rows if r["target"] == t))
             for t in {r["target"] for r in seg_rows}}.items())},
        "per_mode_pool_depth": {name: len(pool[name]) for name in sorted(pool)},
        "adjudication": adjudication,
        "in_subset": in_subset,
        "tfam_modes": tfam_modes,
        "tfam_reasons": tfam_reasons,
        "tfam_skipped_by_depth_gate": tfam_skipped,
        "drop_reasons": dict(drop_reasons),
        "mirror_dedup": mirror_dedup,
        "v1_comparison": {
            "v1_final_n": len(v1_final),
            "family_to_expected_v2_mode": v1_map_expect,
            "status": dict(v1_status),
            "n_new_in_v2": n_new,
            "detail": v1_detail,
        },
        "coverage": {name: {"union": cov_union.get(name, 0),
                            "name_only": cov_name.get(name, 0)}
                     for name in COVERAGE_PATTERNS},
    }

    with open(os.path.join(args.out_dir, "mode_map.json"), "w", encoding="utf-8") as f:
        json.dump(mode_map, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "relabel_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    csv_path = os.path.join(args.out_dir, "relabel_per_segment.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,norm_desc,target,mode_id,ul,direction,confidence,"
                "family_d045,hits\n")
        for r in seg_rows:
            f.write(f"{r['filename']},{r['norm_desc']},{r['target']},"
                    f"{r['mode_id']},{r['ul']},{r['direction']},"
                    f"{r['confidence']},{r['family_d045']},"
                    f"{'|'.join(r['hits'])}\n")

    # 候选池 + 划分 + 泄漏
    quota_table = {}
    for name in in_subset:
        lst = pool[name]
        picked = selections[name]
        role = "test(T-fam)" if name in tfam_modes else "train"
        quota_table[name] = {
            "role": role, "quota": min(12 if name in tfam_modes else 18, len(lst)),
            "pool_depth_after_hardfilter": len(lst),
            "selected": len(picked),
            "actors": len({x["actor_uid"] for x in picked}),
            "tseg_selected": sum(1 for x in picked if x["t_role"] == "T-seg"),
            "dev_selected": sum(1 for x in picked if x["t_role"] == "dev"),
            "confidence_dist": dict(Counter(x["confidence"] for x in picked)),
            "ul_dist": dict(Counter(x["ul"] for x in picked)),
            "direction_dist": dict(Counter(x["direction"] for x in picked)),
            "tfam_actor_conflict_skips": tfam_conflict_skips.get(name, 0),
            "has_npz_already": sum(1 for x in picked if x["has_npz"]),
        }
    candidates = {"_meta": meta_common,
                  "pool_totals": {"selected_all": len(all_selected),
                                  "selected_modes": sum(len(selections[n])
                                                        for n in in_subset),
                                  "intermediate_material": len(inter_sel)},
                  "quota_table": quota_table,
                  "segments": all_selected}
    split = {"_meta": {**meta_common,
                       "set_role_rule": "T-fam mode->test(T-fam)；train mode×bucket9"
                                        "->test(T-seg)；×bucket8->dev；其余->train；"
                                        "intermediate 材料->train(0.3 权重)"},
             "segments": [{k: v for k, v in r.items() if k != "_desc_text"}
                          for r in all_selected]}
    with open(os.path.join(args.out_dir, "candidates_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "split_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "leak_check_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(leak, f, ensure_ascii=False, indent=1)

    # 新段转换清单 + labels（v2 专用目录，不碰 v1 npz/ 与 manifest.json）
    need_conv = [r for r in all_selected if not r["has_npz"]]
    with open(os.path.join(args.out_dir, "csv_list_v2.txt"), "w",
              encoding="utf-8") as f:
        for r in need_conv:
            f.write(os.path.join(DS_DIR, r["move_g1_path"]) + "\n")
    labels = {r["stem"]: {"family": f"v2:{r['target']}", "mode_id": r["mode_id"],
                          "ul": r["ul"], "direction": r["direction"],
                          "label_confidence": r["confidence"],
                          "labels_5dim": {"movement": r["direction"],
                                          "upper_body": r["ul"]},
                          "label_sources": {"filename": r["confidence"] in
                                            ("high", "med"),
                                            "metadata": r["confidence"] == "low",
                                            "manual": False}}
             for r in all_selected}
    with open(os.path.join(args.out_dir, "labels_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)

    # ---------------- 控制台汇总 ----------------
    print(f"[done] targets: {stats['target_dist_segments']}")
    print(f"[done] ul: {stats['ul_dist_segments']}")
    for name, t in quota_table.items():
        print(f"  {name:18s} [{t['role']:12s}] pool={t['pool_depth_after_hardfilter']:6d} "
              f"quota={t['quota']:3d} sel={t['selected']:3d} actors={t['actors']:3d} "
              f"npz_ready={t['has_npz_already']:3d}")
    print(f"  intermediate material: {len(inter_sel)} (w={INTERMEDIATE_WEIGHT})")
    print(f"  v1 comparison: {dict(v1_status)} new={n_new}")
    print(f"  leak pass_all: {leak['pass_all']}  tfam={tfam_modes}")
    print(f"  need conversion: {len(need_conv)}")


if __name__ == "__main__":
    main()
