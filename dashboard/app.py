"""Streamlit monitoring view backed exclusively by FatigueGuard SQLite."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATABASE_PATH, HISTORY_SIZE
from src.database import count_events, get_current_status, get_recent_events, initialize_database


def load_dashboard_data(db_path: Path = DATABASE_PATH) -> tuple[dict | None, list[dict], int]:
    """Read one consistent dashboard snapshot without importing video/model code."""
    initialize_database(db_path)
    return get_current_status(db_path), get_recent_events(100, db_path), count_events(db_path)


def format_probability(value) -> str:
    if value is None:
        return "Sin inferencia"
    return f"{float(value):.3f}"


def main() -> None:
    st.set_page_config(page_title="FatigueGuard Mining", page_icon="🚚", layout="wide")
    st.title("FatigueGuard Mining")
    st.caption("Monitoreo del estado escrito por la aplicación realtime en SQLite")

    if st.button("Actualizar"):
        st.rerun()

    status, events, total_events = load_dashboard_data()
    if status is None:
        st.info("Todavía no existe una sesión. Ejecute la aplicación realtime para iniciar el monitoreo.")
        st.metric("Eventos registrados", total_events)
        return

    state = status["status"]
    if state == "ALERTA":
        st.error("Estado actual: ALERTA")
    elif state == "POSIBLE SOMNOLENCIA":
        st.warning("Estado actual: POSIBLE SOMNOLENCIA")
    elif state == "NORMAL":
        st.success("Estado actual: NORMAL")
    else:
        st.info(f"Estado actual: {state}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equipo", status["equipment_id"])
    col2.metric("Sesión", status["session_id"])
    col3.metric("P(Somnolencia)", format_probability(status["probability"]))
    col4.metric(
        "Positivas recientes",
        f"{status['positive_windows']}/{HISTORY_SIZE}",
    )
    st.caption(f"Última actualización (UTC): {status['last_update']}")

    alert_count = sum(bool(event["alert_triggered"]) for event in events)
    metric1, metric2 = st.columns(2)
    metric1.metric("Eventos registrados", total_events)
    metric2.metric("Alertas entre los últimos 100 eventos", alert_count)

    recent_alerts = [event for event in events if event["alert_triggered"]][:5]
    st.subheader("Alertas recientes")
    if recent_alerts:
        for event in recent_alerts:
            st.warning(
                f"{event['timestamp_start']} — {event['alert_reason']} "
                f"(máx. P={format_probability(event['max_probability'])})"
            )
    else:
        st.write("No hay alertas registradas.")

    st.subheader("Eventos recientes")
    if not events:
        st.write("No hay eventos registrados.")
        return

    table = pd.DataFrame(events).rename(
        columns={
            "timestamp_start": "Inicio (UTC)",
            "timestamp_end": "Fin (UTC)",
            "duration_seconds": "Duración (s)",
            "alert_reason": "Motivo",
            "max_probability": "Probabilidad máxima",
            "mean_perclos": "PERCLOS medio",
            "max_eye_closure_seconds": "Cierre ocular máximo (s)",
        }
    )
    visible_columns = [
        "Inicio (UTC)", "Fin (UTC)", "Duración (s)", "Motivo",
        "Probabilidad máxima", "PERCLOS medio", "Cierre ocular máximo (s)",
    ]
    st.dataframe(table[visible_columns], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

