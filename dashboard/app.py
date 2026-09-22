"""Dashboard operacional de FatigueGuard basado únicamente en SQLite."""

from __future__ import annotations

from datetime import date, datetime, timezone
from html import escape
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATABASE_PATH
from src.database import count_events, get_current_status, get_recent_events, initialize_database


LOCAL_TIMEZONE = ZoneInfo("America/Bogota")

# Realtime escribe varias veces por segundo. Quince segundos toleran pausas breves.
MONITOR_STALE_SECONDS = 15.0

EVENT_COLUMNS = [
    "event_id",
    "equipment_id",
    "timestamp_start",
    "timestamp_end",
    "duration_seconds",
    "alert_reason",
    "max_probability",
    "mean_probability",
    "alert_triggered",
]


def load_dashboard_data(db_path: Path = DATABASE_PATH) -> tuple[dict | None, list[dict], int]:
    """Lee el último estado y todos los eventos sin importar código de cámara o modelo."""
    initialize_database(db_path)
    total_events = count_events(db_path)
    events = get_recent_events(max(1, total_events), db_path)
    return get_current_status(db_path), events, total_events


def to_local_datetime(value) -> datetime | None:
    """Convierte un timestamp almacenado en UTC a la hora de Colombia."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(LOCAL_TIMEZONE).to_pydatetime()


def monitoring_state(
    last_update,
    now: datetime | None = None,
) -> tuple[str, float | None]:
    """Determina si realtime sigue escribiendo, sin crear un estado analítico nuevo."""
    updated = to_local_datetime(last_update)
    if updated is None:
        return "DETENIDO", None

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    elapsed = max(
        0.0,
        (current_time.astimezone(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds(),
    )
    state = "ACTIVO" if elapsed <= MONITOR_STALE_SECONDS else "DETENIDO"
    return state, elapsed


def format_relative_time(elapsed_seconds: float | None) -> str:
    if elapsed_seconds is None:
        return "sin actualizaciones"
    if elapsed_seconds < 60:
        return f"hace {int(elapsed_seconds)} s"
    if elapsed_seconds < 3_600:
        return f"hace {int(elapsed_seconds // 60)} min"
    if elapsed_seconds < 86_400:
        return f"hace {int(elapsed_seconds // 3_600)} h"
    return f"hace {int(elapsed_seconds // 86_400)} días"


def format_local_timestamp(value) -> str:
    local_time = to_local_datetime(value)
    return local_time.strftime("%d/%m/%Y %H:%M:%S") if local_time else "--"


def format_probability(value) -> str:
    if value is None:
        return "--"
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(probability):
        return "--"
    return f"{probability * 100:.1f} %"


def is_alert(value) -> bool:
    """Interpreta el indicador histórico sin depender de su tipo en pandas."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "sí", "si"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def prepare_events(events: list[dict]) -> pd.DataFrame:
    """Normaliza los eventos para filtros, métricas y gráficos."""
    frame = pd.DataFrame(events)
    for column in EVENT_COLUMNS:
        if column not in frame:
            frame[column] = None

    if frame.empty:
        frame["start_local"] = pd.Series(dtype="datetime64[ns, America/Bogota]")
        frame["event_date"] = pd.Series(dtype="object")
        frame["event_hour"] = pd.Series(dtype="Int64")
        frame["is_alert"] = pd.Series(dtype="bool")
        return frame

    frame["start_local"] = pd.to_datetime(
        frame["timestamp_start"], utc=True, errors="coerce"
    ).dt.tz_convert(LOCAL_TIMEZONE)
    frame["end_local"] = pd.to_datetime(
        frame["timestamp_end"], utc=True, errors="coerce"
    ).dt.tz_convert(LOCAL_TIMEZONE)
    frame["event_date"] = frame["start_local"].dt.date
    frame["event_hour"] = frame["start_local"].dt.hour.astype("Int64")
    frame["duration_seconds"] = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    frame["max_probability"] = pd.to_numeric(frame["max_probability"], errors="coerce")
    frame["mean_probability"] = pd.to_numeric(frame["mean_probability"], errors="coerce")
    frame["is_alert"] = frame["alert_triggered"].map(is_alert)
    return frame.sort_values("start_local", ascending=False, na_position="last")


def filter_events(
    events: pd.DataFrame,
    start_date: date,
    end_date: date,
    equipment: str = "Todos",
) -> pd.DataFrame:
    """Aplica solamente los filtros operacionales visibles en el tablero."""
    if events.empty:
        return events.copy()
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    mask = events["event_date"].map(
        lambda value: pd.notna(value) and start_date <= value <= end_date
    )
    if equipment != "Todos":
        mask &= events["equipment_id"].astype(str).eq(equipment)
    return events.loc[mask].copy()


def event_counts(events: pd.DataFrame, selected_date: date | None = None) -> tuple[int, int]:
    selected = events
    if selected_date is not None and not events.empty:
        selected = events[events["event_date"].eq(selected_date)]
    return len(selected), int(selected["is_alert"].sum()) if not selected.empty else 0


def events_by_hour(events: pd.DataFrame) -> pd.DataFrame:
    hours = pd.Index(range(24), name="Hora")
    event_values = events.groupby("event_hour").size().reindex(hours, fill_value=0)
    alert_values = (
        events[events["is_alert"]].groupby("event_hour").size().reindex(hours, fill_value=0)
    )
    return pd.DataFrame({"Eventos": event_values, "Alertas": alert_values})


def events_by_date(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["Eventos", "Alertas"])
    event_values = events.groupby("event_date").size()
    alert_values = events[events["is_alert"]].groupby("event_date").size()
    return pd.DataFrame({"Eventos": event_values, "Alertas": alert_values}).fillna(0).astype(int)


def duration_by_date(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["Duración promedio (s)"])
    values = events.groupby("event_date")["duration_seconds"].mean()
    return values.to_frame("Duración promedio (s)").round(1)


def display_event_table(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Inicio": events["start_local"].map(
                lambda value: value.strftime("%d/%m/%Y %H:%M:%S") if pd.notna(value) else "--"
            ),
            "Equipo": events["equipment_id"].fillna("--").astype(str),
            "Duración": events["duration_seconds"].map(
                lambda value: f"{value:.1f} s" if pd.notna(value) else "--"
            ),
            "Probabilidad máxima": events["max_probability"].map(format_probability),
            "Probabilidad media": events["mean_probability"].map(format_probability),
            "Alerta": events["is_alert"].map({True: "Sí", False: "No"}),
        }
    )


def add_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
        .fg-subtitle {color: #56616f; margin-top: -0.6rem; margin-bottom: 1.4rem;}
        .fg-monitor {border: 1px solid #d9dee5; border-left: 5px solid #687383;
                     border-radius: 8px; padding: 0.9rem 1rem; background: #f7f8fa;}
        .fg-monitor.active {border-left-color: #2e7d32; background: #f4faf5;}
        .fg-monitor strong {font-size: 1.05rem;}
        div[data-testid="stMetric"] {border: 1px solid #e0e4e9; border-radius: 8px;
                                      padding: 0.8rem 1rem; background: white;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_monitoring_header(status: dict | None) -> tuple[str, float | None]:
    last_update = status.get("last_update") if status else None
    monitor_state, elapsed = monitoring_state(last_update)
    css_class = "active" if monitor_state == "ACTIVO" else ""
    updated_text = format_local_timestamp(last_update)
    equipment = escape(str(status.get("equipment_id", "--"))) if status else "--"
    st.markdown(
        f"""
        <div class="fg-monitor {css_class}">
          <strong>Monitoreo: {monitor_state}</strong><br>
          Última actualización: {updated_text} ({format_relative_time(elapsed)})<br>
          Equipo actual: {equipment}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"El monitoreo se considera detenido tras {MONITOR_STALE_SECONDS:.0f} segundos sin actualizaciones."
    )
    return monitor_state, elapsed


def render_charts(events: pd.DataFrame) -> None:
    st.subheader("Comportamiento de los eventos")
    if events.empty:
        st.info("No hay eventos registrados para este periodo.")
        return

    first, second = st.columns(2)
    with first:
        st.markdown("**Eventos y alertas por hora del día**")
        st.bar_chart(events_by_hour(events), color=["#58738f", "#b84949"])
    with second:
        st.markdown("**Eventos y alertas por fecha**")
        st.bar_chart(events_by_date(events), color=["#58738f", "#b84949"])

    st.markdown("**Duración promedio por día**")
    st.bar_chart(duration_by_date(events), color="#7b8794")


def main() -> None:
    st.set_page_config(page_title="FatigueGuard Mining", layout="wide")
    add_dashboard_style()

    title_column, refresh_column = st.columns([5, 1])
    with title_column:
        st.title("FatigueGuard Mining")
        st.markdown(
            '<div class="fg-subtitle">Monitoreo de patrones asociados con somnolencia</div>',
            unsafe_allow_html=True,
        )
    with refresh_column:
        if st.button("Actualizar", use_container_width=True):
            st.rerun()

    status, raw_events, _ = load_dashboard_data()
    events = prepare_events(raw_events)
    render_monitoring_header(status)

    today = datetime.now(LOCAL_TIMEZONE).date()
    today_events, today_alerts = event_counts(events, today)
    last_state = str(status.get("status", "SIN DATOS")) if status else "SIN DATOS"
    current_equipment = str(status.get("equipment_id", "--")) if status else "--"

    st.subheader("Último estado analítico")
    cards = st.columns(4)
    cards[0].metric("Último estado", last_state)
    cards[1].metric("Alertas hoy", today_alerts)
    cards[2].metric("Eventos hoy", today_events)
    cards[3].metric("Equipo monitoreado", current_equipment)

    if status is None:
        st.info("Aún no existe un estado de monitoreo registrado.")

    available_dates = events["event_date"].dropna().tolist()
    minimum_date = min(available_dates) if available_dates else today
    maximum_date = max(available_dates) if available_dates else today
    equipment_options = sorted(
        value for value in events["equipment_id"].dropna().astype(str).unique().tolist()
    )

    st.sidebar.header("Filtros")
    selected_dates = st.sidebar.date_input(
        "Rango de fechas",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    selected_equipment = st.sidebar.selectbox("Equipo", ["Todos", *equipment_options])
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    filtered_events = filter_events(events, start_date, end_date, selected_equipment)
    period_events, period_alerts = event_counts(filtered_events)
    max_probability = filtered_events["max_probability"].max() if not filtered_events.empty else None
    average_duration = filtered_events["duration_seconds"].mean() if not filtered_events.empty else None

    st.subheader("Resumen del periodo")
    duration_text = f"{average_duration:.1f} s" if pd.notna(average_duration) else "--"
    st.write(
        f"**{period_events} eventos** · **{period_alerts} alertas** · "
        f"Probabilidad máxima: **{format_probability(max_probability)}** · "
        f"Duración promedio: **{duration_text}**"
    )

    render_charts(filtered_events)

    st.subheader("Eventos recientes")
    if filtered_events.empty:
        st.info("No hay eventos registrados para este periodo.")
    else:
        st.dataframe(
            display_event_table(filtered_events.head(100)),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Evento: episodio registrado de posible somnolencia. "
            "Alerta: evento que alcanzó el criterio de alerta del sistema."
        )


if __name__ == "__main__":
    main()
