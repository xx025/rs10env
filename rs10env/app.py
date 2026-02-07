"""RS10Env 应用：Streamlit 界面（单棋盘多策略 / 多局对比）。"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

from rs10env.run import run_strategies_on_board, run_strategies, STRATEGY_NAMES


def main() -> None:
    """启动 Streamlit 界面。"""
    import streamlit.web.cli as stcli
    file = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(file), "--server.headless", "true"]
    stcli.main()


def parse_board_text(text: str, H: int, W: int) -> np.ndarray:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != H:
        raise ValueError(f"需要 {H} 行，当前 {len(lines)} 行")
    board = np.zeros((H, W), dtype=np.int32)
    for i, line in enumerate(lines):
        parts = line.replace(",", " ").split()
        if len(parts) != W:
            raise ValueError(f"第 {i+1} 行需要 {W} 个数")
        for j, p in enumerate(parts):
            v = int(p)
            if not 0 <= v <= 9:
                raise ValueError(f"格值须在 0–9")
            board[i, j] = v
    return board


# ---------- Streamlit 页面 ----------
st.set_page_config(page_title="RS10Env 策略结果", page_icon="🎮", layout="centered")
st.title("🎮 RS10Env 策略结果")
st.caption("单棋盘多策略 / 多局对比，最优高亮。")

H, W, target_sum = 16, 10, 10

with st.sidebar:
    st.header("运行参数")
    device = st.selectbox("设备", options=["auto", "cpu", "cuda"], index=0)
    device = None if device == "auto" else device
    selected = st.multiselect(
        "选择策略",
        options=STRATEGY_NAMES,
        default=["random", "greedy", "center_bias", "max_future_moves"],
        help="可多选",
    )

mode = st.radio("模式", ["单棋盘多策略", "多局对比"], horizontal=True)

if mode == "单棋盘多策略":
    st.subheader("棋盘")
    board_source = st.radio("棋盘来源", ["随机生成", "手动输入"], horizontal=True, key="board_src")
    board = None

    if board_source == "随机生成":
        seed = st.number_input("随机种子", min_value=0, value=42, step=1, key="seed")
        if st.button("生成棋盘"):
            rng = np.random.default_rng(int(seed))
            board = rng.integers(1, 10, size=(H, W), dtype=np.int32)
            st.session_state["current_board"] = board
        if "current_board" in st.session_state:
            board = st.session_state["current_board"]
            st.write("当前棋盘：")
            st.dataframe(board, use_container_width=True, height=360)

    if board_source == "手动输入":
        help_text = f"每行 {W} 个数字（0–9），共 {H} 行。"
        default_example = "\n".join(" ".join(str((i + j) % 9 + 1) for j in range(W)) for i in range(H))
        text = st.text_area("棋盘内容", value=default_example, height=320, help=help_text, key="board_text")
        if st.button("解析并预览"):
            try:
                b = parse_board_text(text, H, W)
                st.session_state["current_board"] = b
                st.success("解析成功")
            except Exception as e:
                st.error(str(e))
        if "current_board" in st.session_state:
            board = st.session_state["current_board"]
            st.write("当前棋盘：")
            st.dataframe(board, use_container_width=True, height=360)

    run_clicked = st.button("运行各策略", type="primary", key="run_single")
    if run_clicked and board_source == "手动输入":
        try:
            board = parse_board_text(text, H, W)
            st.session_state["current_board"] = board
        except Exception as e:
            st.error(f"棋盘解析失败：{e}")
            st.stop()
    if board is None and "current_board" in st.session_state:
        board = st.session_state["current_board"]

    if not selected:
        st.warning("请至少选择一条策略。")
        st.stop()
    if board is None and run_clicked:
        st.warning("请先生成或解析棋盘再运行。")
        st.stop()

    if run_clicked and board is not None:
        with st.spinner("正在运行各策略..."):
            try:
                results = run_strategies_on_board(
                    board=board, strategy_names=selected, device=device,
                    H=H, W=W, target_sum=target_sum,
                )
            except Exception as e:
                st.error(str(e))
                st.stop()
        rows_html = []
        for r in results:
            row = (
                "<tr style=\"background: {}; font-weight: {};\">"
                "<td>{}</td><td>{:.4f}</td><td>{}</td><td>{}</td><td>{}</td></tr>"
            ).format(
                "#d4edda" if r["is_best"] else "transparent",
                "bold" if r["is_best"] else "normal",
                r["strategy_name"], r["total_reward"], r["steps"], r["total_cleared"],
                "★ 最优" if r["is_best"] else "",
            )
            rows_html.append(row)
        st.markdown(
            "<table style='width:100%; border-collapse: collapse;'>"
            "<thead><tr style='border-bottom: 2px solid #ddd;'>"
            "<th style='text-align:left'>策略</th><th style='text-align:right'>总奖励</th>"
            "<th style='text-align:right'>步数</th><th style='text-align:right'>消除格数</th><th>备注</th>"
            "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>",
            unsafe_allow_html=True,
        )
        st.success("按「消除格数」判定最优（绿色为最优）。")
    elif board is not None:
        st.info("选择策略后点击「运行各策略」。")

else:
    st.subheader("多局对比")
    num_games = st.number_input("每策略局数", min_value=1, max_value=500, value=20, step=5, key="num_games")
    base_seed = st.number_input("随机种子", min_value=0, value=42, step=1, key="base_seed")
    run_compare = st.button("运行对比", type="primary", key="run_compare")

    if not selected:
        st.warning("请至少选择一条策略。")
        st.stop()

    if run_compare:
        progress_bar = st.progress(0, text="准备中...")
        def on_progress(current: int, total: int, message: str) -> None:
            progress_bar.progress(current / total if total else 0, text=f"已完成 {current}/{total} · {message}")
        try:
            summary = run_strategies(
                strategy_names=selected, num_games=num_games, device=device,
                base_seed=base_seed, H=H, W=W, target_sum=target_sum,
                progress_callback=on_progress,
            )
        except Exception as e:
            progress_bar.empty()
            st.error(str(e))
            st.stop()
        progress_bar.progress(1.0, text="完成")
        progress_bar.empty()

        rows_html = []
        for s in summary:
            row = (
                "<tr style=\"background: {}; font-weight: {};\">"
                "<td>{}</td><td>{:.4f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.4f}s</td><td>{}</td></tr>"
            ).format(
                "#d4edda" if s["is_best"] else "transparent",
                "bold" if s["is_best"] else "normal",
                s["strategy_name"], s["avg_reward"], s["avg_steps"], s["avg_cleared"], s["avg_time_sec"],
                "★ 最优" if s["is_best"] else "",
            )
            rows_html.append(row)
        st.markdown(
            "<table style='width:100%; border-collapse: collapse;'>"
            "<thead><tr style='border-bottom: 2px solid #ddd;'>"
            "<th style='text-align:left'>策略</th><th style='text-align:right'>平均奖励</th>"
            "<th>平均步数</th><th>平均消除</th><th>平均时间</th><th>备注</th>"
            "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>",
            unsafe_allow_html=True,
        )
        st.success(f"共 {num_games} 局 × {len(selected)} 条策略，按「平均消除」判定最优。")
    else:
        st.info("设置局数与种子后点击「运行对比」。")
