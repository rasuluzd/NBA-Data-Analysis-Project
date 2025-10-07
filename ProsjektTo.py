from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE_DIR = Path(__file__).resolve().parent
DATA_KATALOG = BASE_DIR / "data"
CSV_FILSTI = DATA_KATALOG / "NBA_Player_Stats.csv"
DB_FILSTI = DATA_KATALOG / "nba_player_stats.sqlite"

STATISTIKK_VALG = {
    "P": ("Total Poeng", "total_points"),
    "PP": ("Poeng per Kamp", "ppg"),
    "A": ("Målgivende Pasninger per Kamp", "apg"),
    "T": ("True Shooting Prosent", "ts"),
    "3": ("3-Poeng Forsøk", "three_pa"),
}


def sesong_nokkel(season: str) -> int:
    start = season.split("-")[0]
    if len(start) == 4:
        return int(start)
    value = int(start)
    return 1900 + value if value >= 50 else 2000 + value


def les_spillerdata(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Les NBA-statistikk med pandas og forbered både rå- og totalsesongdata."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fant ikke {csv_path}. Plasser CSV-filen i data/-mappen."
        )

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df["Player_lower"] = df["Player"].str.lower()
    df["Season"] = df["Season"].astype(str)
    numeric_cols = ["PTS", "AST", "TRB", "3PA", "3P%", "FGA", "FTA", "G"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TS_calc"] = beregn_true_shooting(df)

    season_df = prioriter_tot_rader(df)
    seasons = sorted(season_df["Season"].unique(), key=sesong_nokkel)
    return df, season_df, seasons


def beregn_true_shooting(df: pd.DataFrame) -> pd.Series:
    denominator = df["FGA"].astype(float) + 0.44 * df["FTA"].astype(float)
    denominator = denominator.replace({0: np.nan})
    ts = df["PTS"].astype(float) / (2 * denominator)
    return ts.fillna(0.0)


def prioriter_tot_rader(df: pd.DataFrame) -> pd.DataFrame:
    """Lever én rad per spiller/sesong, der TOT prioriteres ved lagbytter."""
    working = df.copy()
    working["tm_priority"] = np.where(working["Tm"] == "TOT", 0, 1)
    working.sort_values(["Player_lower", "Season", "tm_priority"], inplace=True)
    season_df = working.drop_duplicates(subset=["Player_lower", "Season"], keep="first")
    season_df.drop(columns=["tm_priority"], inplace=True)
    return season_df.reset_index(drop=True)


def lagre_til_sqlite(df: pd.DataFrame, db_path: Path, table_name: str = "player_seasons") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def skriv_hurtiginnsikt(sesong_df: pd.DataFrame) -> None:
    sesonger = sorted(sesong_df["Season"].unique(), key=sesong_nokkel)
    forste_sesong, siste_sesong = sesonger[0], sesonger[-1]
    unike_spillere = sesong_df["Player_lower"].nunique()
    print("\n Hurtiginnsikt")
    print(
        f"  • Datasettet dekker {forste_sesong} til {siste_sesong} med {len(sesonger)} sesonger og {unike_spillere} unike spillere."
    )

    top_scorers = sesong_df.dropna(subset=["PTS"]).nlargest(5, "PTS")
    if not top_scorers.empty:
        print("  • Mestscorende sesonger (poeng per kamp):")
        for _, row in top_scorers.iterrows():
            print(
                f"      - {row['Player']} ({row['Season']}): {float(row['PTS']):.1f} PPG"
            )

    top_assists = sesong_df.dropna(subset=["AST"]).nlargest(5, "AST")
    if not top_assists.empty:
        print("  • Beste playmakere (assist per kamp):")
        for _, row in top_assists.iterrows():
            print(
                f"      - {row['Player']} ({row['Season']}): {float(row['AST']):.1f} APG"
            )

    top_shooters = sorter_tre(sesong_df).head(5)
    if not top_shooters.empty:
        print("  • Trefarlige sesonger (treff per kamp):")
        for _, row in top_shooters.iterrows():
            makes = float(row["three_makes_per_game"])
            print(
                f"      - {row['Player']} ({row['Season']}): {makes:.2f} 3PM"
            )
    print("  • Tips: Skriv 'liste' når du blir bedt om spillernavn for flere forslag.\n")


def kjor_sql_demo(db_path: Path, limit: int = 5) -> None:
    try:
        with sqlite3.connect(db_path) as conn:
            demo = pd.read_sql_query(
                "SELECT Player, Season, ROUND(PTS, 1) AS PPG FROM player_seasons ORDER BY PTS DESC LIMIT ?",
                conn,
                params=(limit,),
            )
    except Exception as exc:  # pragma: no cover - beskytt mot miljøfeil
        print(f" Klarte ikke å lese fra databasen: {exc}")
        return

    if not demo.empty:
        print("Eksempel fra databasen (lagret for videre analyser):")
        print(demo.to_string(index=False))
        print()


def _unike_topprader(df: pd.DataFrame, metrikk: str, topp_n: int) -> List[pd.Series]:
    rader: List[pd.Series] = []
    sett_spillere: set[str] = set()
    for _, rad in df.sort_values(metrikk, ascending=False).iterrows():
        nøkkel = rad.get("Player_lower")
        if not isinstance(nøkkel, str):
            continue
        if nøkkel in sett_spillere or pd.isna(rad.get(metrikk)):
            continue
        rader.append(rad)
        sett_spillere.add(nøkkel)
        if len(rader) == topp_n:
            break
    return rader


def beregn_spesialister(sesong_df: pd.DataFrame, topp_n: int = 5) -> Dict[str, List[Dict[str, float]]]:
    df = sesong_df.copy()
    df["fgm_total"] = df["FGA"] * df["FG%"] * df["G"]

    poengledere = _unike_topprader(df.dropna(subset=["PTS"]), "PTS", topp_n)
    playmakere = _unike_topprader(df.dropna(subset=["AST"]), "AST", topp_n)

    skyttere_df = sorter_tre(df)
    treskyttere = _unike_topprader(skyttere_df, "three_makes_per_game", topp_n)

    ts_kandidater = df[(df["fgm_total"] >= 300) & df["TS_calc"].notna()]
    ts_spesialister = _unike_topprader(ts_kandidater, "TS_calc", topp_n)

    def _formater_poster(rader: List[pd.Series], metrikk: str, faktor: float = 1.0) -> List[Dict[str, float]]:
        poster: List[Dict[str, float]] = []
        for rad in rader:
            poster.append(
                {
                    "spiller": str(rad["Player"]),
                    "sesong": str(rad["Season"]),
                    "verdi": float(rad[metrikk]) * faktor,
                }
            )
        return poster

    return {
        "poengledere": _formater_poster(poengledere, "PTS"),
        "playmakere": _formater_poster(playmakere, "AST"),
        "treskyttere": _formater_poster(treskyttere, "three_makes_per_game"),
        "true_shooting": _formater_poster(ts_spesialister, "TS_calc", faktor=100.0),
    }


def vis_spillertips(sesong_df: pd.DataFrame) -> None:
    spesialister = beregn_spesialister(sesong_df)
    print("Forslag til spesialister du kan utforske:")

    def _skriv_kategori(tittel: str, poster: List[Dict[str, float]], enhet: str) -> None:
        if not poster:
            return
        print(f"  {tittel}:")
        for post in poster:
            print(f"    - {post['spiller']} ({post['sesong']}): {post['verdi']:.2f} {enhet}")

    _skriv_kategori("Poengledere", spesialister["poengledere"], "poeng per kamp")
    _skriv_kategori("Playmakere", spesialister["playmakere"], "assist per kamp")
    _skriv_kategori("Trepoengsskyttere", spesialister["treskyttere"], "treff per kamp")
    _skriv_kategori("True shooting-ledere", spesialister["true_shooting"], "TS%")
    print()


def spillerdatasett(sesong_df: pd.DataFrame, spiller: str) -> pd.DataFrame:
    mask = sesong_df["Player_lower"] == spiller.lower()
    return (
        sesong_df.loc[mask]
        .sort_values("Season", key=lambda kolonne: kolonne.map(sesong_nokkel))
        .copy()
    )


def bygg_statistikk_tabell(spiller_df: pd.DataFrame) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for _, rad in spiller_df.iterrows():
        kamper = float(rad["G"])
        record = {
            "season": rad["Season"],
            "total_points": int(round(kamper * float(rad["PTS"]))),
            "ppg": float(rad["PTS"]),
            "apg": float(rad["AST"]),
            "ts": float(rad["TS_calc"]) * 100,
            "three_pa": float(rad["3PA"]),
            "three_pct": float(rad.get("3P%", 0.0)) * 100,
        }
        records.append(record)
    return records


def plott_spillere(
    sesong_df: pd.DataFrame,
    sesonger: Iterable[str],
    spillere: List[str],
    statistikk_nokkel: str,
) -> None:
    etikett, metrikk_nokkel = STATISTIKK_VALG[statistikk_nokkel]
    fig, akser = plt.subplots(2, 1, figsize=(10, 8))

    sesongliste = list(sesonger)
    for akse in akser:
        akse.plot(sesongliste, np.zeros(len(sesongliste)), alpha=0)

    for navn in spillere:
        spiller_df = spillerdatasett(sesong_df, navn)
        if spiller_df.empty:
            print(f"Fant ingen sesonger for {navn}.")
            continue
        statistikk = bygg_statistikk_tabell(spiller_df)
        x = np.arange(len(statistikk))
        y = [rad[metrikk_nokkel] for rad in statistikk]
        sesongetiketter = [rad["season"] for rad in statistikk]

        akser[0].plot(sesongetiketter, y, marker="o", label=navn.title())

        if statistikk_nokkel == "3":
            prosentverdier = [rad["three_pct"] for rad in statistikk]
            akser[1].plot(sesongetiketter, prosentverdier, marker="s", label=f"{navn.title()} 3P%")
            akser[1].set_ylabel("3-Poeng Prosent")
            akser[1].set_title("3-poeng prosent")
        else:
            if len(y) >= 4:
                try:
                    parametre, _ = curve_fit(polynom_4, x, y)
                    modellert = polynom_4(x, *parametre)
                    akser[1].plot(sesongetiketter, modellert, label=f"{navn.title()} modell")
                except Exception as exc:  # pragma: no cover - kun for plotting
                    print(f"Kunne ikke tilpasse modell for {navn}: {exc}")
                    akser[1].plot(sesongetiketter, y, linestyle="--", label=f"{navn.title()} trend")
            else:
                akser[1].plot(sesongetiketter, y, linestyle="--", label=f"{navn.title()} trend")
                akser[1].set_title("Trend (for få datapunkter for regresjon)")

    for akse in akser:
        akse.set_xticks(sesongliste)
        akse.set_xticklabels(sesongliste, rotation=90, fontsize=7)
        akse.set_xlabel("Sesong")
        akse.legend()

    akser[0].set_title("Spillerstatistikk")
    akser[0].set_ylabel(etikett)
    if statistikk_nokkel != "3":
        akser[1].set_ylabel(etikett)
        akser[1].set_title("Regresjonsmodell")
    else:
        akser[1].set_ylabel("3-Poeng Prosent")

    fig.tight_layout()
    plt.show()


def polynom_4(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def sporr_statistikkvalg() -> str:
    print("Hvilken statistikk vil du plotte?")
    for kode, (etikett, _) in STATISTIKK_VALG.items():
        print(f"  {kode}: {etikett}")
    print("Tips: Velg 'P' for totalpoeng, eller '3' for å se trepoengsforsøk og treff.")
    while True:
        valg = input("Velg (P, PP, A, T eller 3): ").strip().upper()
        if valg in STATISTIKK_VALG:
            return valg
        print("Ugyldig valg. Prøv igjen.")


def sporr_spillere(sesong_df: pd.DataFrame) -> List[str]:
    gyldige_spillere = set(sesong_df["Player_lower"])
    while True:
        try:
            antall = int(input("Hvor mange spillere vil du sammenligne? (1-5): "))
            if 1 <= antall <= 5:
                break
            print("Velg et tall mellom 1 og 5.")
        except ValueError:
            print("Skriv inn et heltall.")

    valgte: List[str] = []
    print("Tips: Skriv 'liste' eller 'hjelp' for forslag til spillere.")
    hjelpebegrep = {"liste", "help", "hjelp", "tips"}
    for _ in range(antall):
        navn = input("Navn på spiller: ").strip().lower()
        while True:
            if navn in hjelpebegrep:
                vis_spillertips(sesong_df)
                navn = input("Velg spiller ved å skrive navnet: ").strip().lower()
                continue
            if navn not in gyldige_spillere:
                print(f"Fant ikke {navn.title()} i datasettet. Skriv 'liste' for forslag eller prøv på nytt.")
                navn = input("Navn på spiller: ").strip().lower()
                continue
            if navn in valgte:
                print("Spilleren er allerede valgt.")
                navn = input("Navn på spiller: ").strip().lower()
                continue
            break
        valgte.append(navn)
    return valgte


def sorter_tre(season_df: pd.DataFrame) -> pd.DataFrame:
    df = season_df.copy()
    df = df[(df["G"] > 30) & df["3P%"].notnull()]
    df["three_makes_per_game"] = df["3PA"] * df["3P%"]
    return df.sort_values("three_makes_per_game", ascending=False).reset_index(drop=True)


def tre_siden_2016(sesong_df: pd.DataFrame) -> None:
    topp = sorter_tre(sesong_df).head(100)
    etter_2016 = topp[
        topp["Season"].apply(lambda s: sesong_nokkel(str(s)) > 2015)
    ]
    print(
        "Av de topp 100 sesongene for 3-poeng treff per kamp, kom"
        f" {len(etter_2016)} etter 2015-16 sesongen."
    )


def trept_teller(sesong_df: pd.DataFrame) -> None:
    sesonger = sorted(sesong_df["Season"].unique(), key=sesong_nokkel)
    andeler = []
    for sesong in sesonger:
        utsnitt = sesong_df[sesong_df["Season"] == sesong]
        store = utsnitt[utsnitt["Pos"].isin(["C", "PF"])]
        antall = len(store)
        terskel = (store["3PA"] > 1.2).sum()
        andel = (terskel / antall * 100) if antall else 0.0
        andeler.append(andel)

    plt.figure(figsize=(10, 4))
    plt.plot(sesonger, andeler, marker="o", label="Andel C/PF med >1.2 3PA per kamp")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Andel (%)")
    plt.xlabel("Sesong")
    plt.title("Utvikling i 3-poeng volum for store spillere")
    plt.legend()
    plt.tight_layout()
    plt.show()


def karriere_3(
    sesong_df: pd.DataFrame,
    spiller_a: str = "Stephen Curry",
    spiller_b: str = "Ray Allen",
) -> None:
    def _aggregert(spiller: str) -> tuple[float, float]:
        rader = sesong_df[sesong_df["Player_lower"] == spiller.lower()]
        treff = float((rader["3PA"] * rader["3P%"] * rader["G"]).sum())
        forsok = float((rader["3PA"] * rader["G"]).sum())
        return treff, forsok

    treff_a, forsok_a = _aggregert(spiller_a)
    treff_b, forsok_b = _aggregert(spiller_b)

    if forsok_a == 0 or forsok_b == 0:
        print("Kan ikke beregne karriereprosent uten forsøk.")
        return

    maal_prosent = treff_b / forsok_b
    bom = 0
    prosent_a = treff_a / forsok_a
    while prosent_a > maal_prosent:
        bom += 1
        prosent_a = treff_a / (forsok_a + bom)

    print(
        f"{spiller_a} kan bomme de neste {max(bom - 1, 0)} treerne sine på rad og fortsatt"
        f" ha høyere 3-poeng prosent enn {spiller_b}."
    )



_, sesong_df, sesonger = les_spillerdata(CSV_FILSTI)
lagre_til_sqlite(sesong_df, DB_FILSTI)
print(f"Data persistert til {DB_FILSTI.resolve()}")

skriv_hurtiginnsikt(sesong_df)
kjor_sql_demo(DB_FILSTI)

statistikknokkel = sporr_statistikkvalg()
spillere = sporr_spillere(sesong_df)
plott_spillere(sesong_df, sesonger, spillere, statistikknokkel)

trept_teller(sesong_df)
tre_siden_2016(sesong_df)
karriere_3(sesong_df)


