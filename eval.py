import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ─── Configuração ────────────────────────────────────────────────────
CAMPOS_OBRIGATORIOS = [
    "marca", "modelo", "tipo_produto", "capacidade_volume",
    "voltagem", "cor", "material", "quantidade", "outros_atributos"
]

CAMPOS_CRITICOS = ["tipo_produto", "marca", "capacidade_volume"]


# ═══════════════════════════════════════════════════════════════════
#  1. QUALIDADE DA EXTRAÇÃO JSON
# ═══════════════════════════════════════════════════════════════════

def avaliar_qualidade_json(df: pd.DataFrame) -> dict:
    """
    Mede a taxa de sucesso/falha da extração do LLM.
    Inspirado em Narayan et al. — FMs são frágeis a variações de prompt,
    então medir a taxa de falha é o primeiro diagnóstico.
    """
    total = len(df)
    resultados = {
        "total_registros": total,
        "json_vazio_ou_erro": 0,
        "json_todos_nulos": 0,
        "json_valido_com_dados": 0,
    }

    for _, row in df.iterrows():
        try:
            j = json.loads(row["json_limpo_python"]) if isinstance(row["json_limpo_python"], str) else row["json_limpo_python"]
        except (json.JSONDecodeError, TypeError):
            resultados["json_vazio_ou_erro"] += 1
            continue

        if not j or all(v is None or v == "" or v == "null" for v in j.values()):
            resultados["json_todos_nulos"] += 1
        else:
            resultados["json_valido_com_dados"] += 1

    resultados["taxa_extracao_ok"] = round(resultados["json_valido_com_dados"] / total * 100, 2)
    resultados["taxa_falha_total"] = round(
        (resultados["json_vazio_ou_erro"] + resultados["json_todos_nulos"]) / total * 100, 2
    )
    return resultados


# ═══════════════════════════════════════════════════════════════════
#  2. COMPLETUDE DE ATRIBUTOS (por campo)
# ═══════════════════════════════════════════════════════════════════

def avaliar_completude_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Taxa de preenchimento por campo.
    Inspirado em Brinkmann et al. (WDC-PAVE) — a normalização de
    atributos depende de primeiro extraí-los corretamente.
    """
    stats = []
    for campo in CAMPOS_OBRIGATORIOS:
        preenchido = 0
        total_valido = 0
        for _, row in df.iterrows():
            try:
                j = json.loads(row["json_limpo_python"]) if isinstance(row["json_limpo_python"], str) else row["json_limpo_python"]
            except (json.JSONDecodeError, TypeError):
                continue
            if not j:
                continue
            total_valido += 1
            val = j.get(campo)
            if val is not None and val != "" and val != "null":
                preenchido += 1

        taxa = round(preenchido / total_valido * 100, 2) if total_valido > 0 else 0.0
        critico = "⚠️ CRÍTICO" if campo in CAMPOS_CRITICOS and taxa < 50 else ""
        stats.append({
            "campo": campo,
            "preenchido": preenchido,
            "total_valido": total_valido,
            "taxa_preenchimento_%": taxa,
            "alerta": critico
        })

    return pd.DataFrame(stats).sort_values("taxa_preenchimento_%", ascending=False)


# ═══════════════════════════════════════════════════════════════════
#  3. ANÁLISE DE AGRUPAMENTO (Clustering)
# ═══════════════════════════════════════════════════════════════════

def avaliar_agrupamento(df: pd.DataFrame) -> dict:
    """
    Mede a capacidade do pipeline de agrupar títulos equivalentes.
    Baseado no framework WDC Products — a qualidade do matcher se
    reflete na distribuição dos clusters resultantes.

    Singletons (grupo com 1 título) são potenciais falhas de
    normalização — o pipeline não conseguiu generalizar.
    """
    nomes = df["nome_canonico_final"].dropna()
    nomes = nomes[nomes.str.strip() != ""]

    contagem = Counter(nomes)
    tamanhos = list(contagem.values())

    total_nomes_unicos = len(contagem)
    total_registros = len(nomes)
    singletons = sum(1 for t in tamanhos if t == 1)
    grupos_2_plus = sum(1 for t in tamanhos if t >= 2)

    resultado = {
        "total_registros_com_nome": total_registros,
        "nomes_canonicos_unicos": total_nomes_unicos,
        "taxa_reducao_%": round((1 - total_nomes_unicos / total_registros) * 100, 2) if total_registros > 0 else 0,
        "singletons": singletons,
        "taxa_singletons_%": round(singletons / total_nomes_unicos * 100, 2) if total_nomes_unicos > 0 else 0,
        "grupos_com_2+_membros": grupos_2_plus,
        "maior_grupo_tamanho": max(tamanhos) if tamanhos else 0,
        "maior_grupo_nome": max(contagem, key=contagem.get) if contagem else "",
        "mediana_tamanho_grupo": int(np.median(tamanhos)) if tamanhos else 0,
    }

    # Top 10 maiores grupos
    resultado["top10_grupos"] = contagem.most_common(10)

    return resultado


# ═══════════════════════════════════════════════════════════════════
#  4. CONSISTÊNCIA INTRA-CATEGORIA
# ═══════════════════════════════════════════════════════════════════

def avaliar_consistencia_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica se o pipeline normaliza de forma consistente dentro
    de cada categoria do MeLi.

    Inspirado em WDC Products (dimensão corner-cases) — produtos da
    mesma categoria que recebem nomes canônicos muito diferentes podem
    indicar inconsistência do LLM.
    """
    stats = []
    for cat, grupo in df.groupby("categoria_original"):
        nomes = grupo["nome_canonico_final"].dropna()
        nomes = nomes[nomes.str.strip() != ""]
        total = len(nomes)
        unicos = nomes.nunique()

        # tipo_produto dominante
        tipos = []
        for _, row in grupo.iterrows():
            try:
                j = json.loads(row["json_limpo_python"]) if isinstance(row["json_limpo_python"], str) else row["json_limpo_python"]
                tp = j.get("tipo_produto") if j else None
                if tp and tp != "null":
                    tipos.append(tp)
            except (json.JSONDecodeError, TypeError):
                continue

        tipo_counter = Counter(tipos)
        tipo_dominante = tipo_counter.most_common(1)[0] if tipo_counter else ("—", 0)
        consistencia_tipo = round(tipo_dominante[1] / len(tipos) * 100, 2) if tipos else 0

        stats.append({
            "categoria": cat,
            "titulos": total,
            "nomes_unicos": unicos,
            "taxa_reducao_%": round((1 - unicos / total) * 100, 2) if total > 0 else 0,
            "tipo_produto_dominante": tipo_dominante[0],
            "consistencia_tipo_%": consistencia_tipo,
        })

    return pd.DataFrame(stats).sort_values("taxa_reducao_%", ascending=False)


# ═══════════════════════════════════════════════════════════════════
#  5. DETECÇÃO DE CORNER-CASES
# ═══════════════════════════════════════════════════════════════════

def detectar_corner_cases(df: pd.DataFrame) -> dict:
    """
    Identifica padrões problemáticos — inspirado na dimensão de
    corner-cases do WDC Products benchmark.

    Corner-cases negativos: títulos diferentes que geraram o mesmo
    nome canônico (possíveis falsos positivos).

    Corner-cases positivos: títulos muito similares que geraram
    nomes canônicos diferentes (possíveis falsos negativos).
    """
    from difflib import SequenceMatcher

    # Nomes canônicos com múltiplos títulos → verificar se são realmente iguais
    contagem = Counter(df["nome_canonico_final"].dropna())
    grupos_grandes = {k: v for k, v in contagem.items() if v >= 3 and k.strip() != ""}

    possiveis_falsos_positivos = []
    for nome, count in sorted(grupos_grandes.items(), key=lambda x: -x[1])[:5]:
        titulos = df[df["nome_canonico_final"] == nome]["titulo_original"].tolist()
        possiveis_falsos_positivos.append({
            "nome_canonico": nome,
            "quantidade": count,
            "exemplos_titulos": titulos[:5]
        })

    # Títulos muito parecidos com nomes canônicos diferentes
    possiveis_falsos_negativos = []
    titulos = df[["titulo_original", "nome_canonico_final"]].dropna().values.tolist()

    # Amostra para não explodir em O(n²)
    sample_size = min(len(titulos), 100)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(titulos), size=sample_size, replace=False)
    sample = [titulos[i] for i in indices]

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            t1, n1 = sample[i]
            t2, n2 = sample[j]
            if n1 != n2:
                sim = SequenceMatcher(None, t1.lower(), t2.lower()).ratio()
                if sim > 0.75:
                    possiveis_falsos_negativos.append({
                        "titulo_1": t1,
                        "titulo_2": t2,
                        "similaridade": round(sim, 3),
                        "canonico_1": n1,
                        "canonico_2": n2,
                    })

    possiveis_falsos_negativos.sort(key=lambda x: -x["similaridade"])

    return {
        "possiveis_falsos_positivos": possiveis_falsos_positivos,
        "possiveis_falsos_negativos": possiveis_falsos_negativos[:10],
    }


# ═══════════════════════════════════════════════════════════════════
#  6. ANÁLISE DA CAMADA SIMBÓLICA (regex vs. LLM)
# ═══════════════════════════════════════════════════════════════════

def avaliar_correcoes_simbolicas(df: pd.DataFrame) -> dict:
    """
    Mede quantas vezes a camada de regex corrigiu a saída do LLM.
    Inspirado no Ditto (span normalization) — se o regex corrige
    muito, o prompt pode ser melhorado.
    """
    total = 0
    correcoes = 0
    campos_corrigidos = Counter()

    for _, row in df.iterrows():
        try:
            bruto = json.loads(row["json_llm_bruto"]) if isinstance(row["json_llm_bruto"], str) else row["json_llm_bruto"]
            limpo = json.loads(row["json_limpo_python"]) if isinstance(row["json_limpo_python"], str) else row["json_limpo_python"]
        except (json.JSONDecodeError, TypeError):
            continue

        if not bruto or not limpo:
            continue

        total += 1
        houve_correcao = False
        for campo in CAMPOS_OBRIGATORIOS:
            val_bruto = str(bruto.get(campo, "")).strip().lower()
            val_limpo = str(limpo.get(campo, "")).strip().lower()

            # Ignora diferenças None/null/"" vs None
            if val_bruto in ("none", "null", "") and val_limpo in ("none", "null", ""):
                continue
            if val_bruto != val_limpo:
                houve_correcao = True
                campos_corrigidos[campo] += 1

        if houve_correcao:
            correcoes += 1

    return {
        "total_analisado": total,
        "registros_com_correcao": correcoes,
        "taxa_correcao_%": round(correcoes / total * 100, 2) if total > 0 else 0,
        "correcoes_por_campo": dict(campos_corrigidos.most_common()),
    }


# ═══════════════════════════════════════════════════════════════════
#  7. RELATÓRIO CONSOLIDADO
# ═══════════════════════════════════════════════════════════════════

def gerar_relatorio(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Executa todas as avaliações e imprime um relatório consolidado."""

    relatorio = {}

    # ── 1. Qualidade JSON ──
    qj = avaliar_qualidade_json(df)
    relatorio["qualidade_json"] = qj
    if verbose:
        print("=" * 65)
        print("  1. QUALIDADE DA EXTRAÇÃO JSON")
        print("=" * 65)
        print(f"  Total de registros:        {qj['total_registros']}")
        print(f"  JSONs válidos com dados:   {qj['json_valido_com_dados']}  ({qj['taxa_extracao_ok']}%)")
        print(f"  JSONs vazios/erro:         {qj['json_vazio_ou_erro']}")
        print(f"  JSONs todos nulos:         {qj['json_todos_nulos']}")
        print(f"  Taxa de falha total:       {qj['taxa_falha_total']}%")
        print()

    # ── 2. Completude de atributos ──
    ca = avaliar_completude_atributos(df)
    relatorio["completude_atributos"] = ca
    if verbose:
        print("=" * 65)
        print("  2. COMPLETUDE DE ATRIBUTOS")
        print("=" * 65)
        for _, row in ca.iterrows():
            barra = "█" * int(row["taxa_preenchimento_%"] / 5) + "░" * (20 - int(row["taxa_preenchimento_%"] / 5))
            alerta = f"  {row['alerta']}" if row['alerta'] else ""
            print(f"  {row['campo']:<22s} {barra} {row['taxa_preenchimento_%']:>6.1f}%  ({row['preenchido']}/{row['total_valido']}){alerta}")
        print()

    # ── 3. Agrupamento ──
    ag = avaliar_agrupamento(df)
    relatorio["agrupamento"] = ag
    if verbose:
        print("=" * 65)
        print("  3. ANÁLISE DE AGRUPAMENTO")
        print("=" * 65)
        print(f"  Registros com nome:        {ag['total_registros_com_nome']}")
        print(f"  Nomes canônicos únicos:    {ag['nomes_canonicos_unicos']}")
        print(f"  Taxa de redução:           {ag['taxa_reducao_%']}%")
        print(f"  Singletons:                {ag['singletons']}  ({ag['taxa_singletons_%']}% dos nomes)")
        print(f"  Grupos com 2+ membros:     {ag['grupos_com_2+_membros']}")
        print(f"  Maior grupo:               {ag['maior_grupo_tamanho']}x  →  \"{ag['maior_grupo_nome'][:60]}\"")
        print(f"  Mediana do tamanho:         {ag['mediana_tamanho_grupo']}")
        print()
        print("  Top 10 maiores grupos:")
        for nome, count in ag["top10_grupos"]:
            print(f"    {count:>3d}x  {nome[:70]}")
        print()

    # ── 4. Consistência por categoria ──
    cc = avaliar_consistencia_por_categoria(df)
    relatorio["consistencia_categoria"] = cc
    if verbose:
        print("=" * 65)
        print("  4. CONSISTÊNCIA POR CATEGORIA")
        print("=" * 65)
        for _, row in cc.iterrows():
            print(f"  {row['categoria'][:40]:<42s} {row['titulos']:>3d} títulos → {row['nomes_unicos']:>3d} nomes  (redução {row['taxa_reducao_%']:>5.1f}%)  tipo: {row['tipo_produto_dominante'][:25]} ({row['consistencia_tipo_%']}%)")
        print()

    # ── 5. Corner-cases ──
    corner = detectar_corner_cases(df)
    relatorio["corner_cases"] = corner
    if verbose:
        print("=" * 65)
        print("  5. CORNER-CASES DETECTADOS")
        print("=" * 65)
        print()
        print("  Possíveis falsos positivos (títulos diferentes → mesmo canônico):")
        if corner["possiveis_falsos_positivos"]:
            for fp in corner["possiveis_falsos_positivos"]:
                print(f"    Nome: \"{fp['nome_canonico'][:60]}\"  ({fp['quantidade']}x)")
                for t in fp["exemplos_titulos"][:3]:
                    print(f"      • {t[:80]}")
                print()
        else:
            print("    Nenhum grupo grande detectado.\n")

        print("  Possíveis falsos negativos (títulos similares → canônicos diferentes):")
        if corner["possiveis_falsos_negativos"]:
            for fn in corner["possiveis_falsos_negativos"][:5]:
                print(f"    Sim={fn['similaridade']:.3f}")
                print(f"      T1: {fn['titulo_1'][:70]}")
                print(f"      T2: {fn['titulo_2'][:70]}")
                print(f"      C1: {fn['canonico_1'][:60]}")
                print(f"      C2: {fn['canonico_2'][:60]}")
                print()
        else:
            print("    Nenhum par similar com canônicos divergentes encontrado.\n")

    # ── 6. Correções simbólicas ──
    cs = avaliar_correcoes_simbolicas(df)
    relatorio["correcoes_simbolicas"] = cs
    if verbose:
        print("=" * 65)
        print("  6. CORREÇÕES DA CAMADA SIMBÓLICA (Regex vs. LLM)")
        print("=" * 65)
        print(f"  Registros analisados:       {cs['total_analisado']}")
        print(f"  Com pelo menos 1 correção:  {cs['registros_com_correcao']}  ({cs['taxa_correcao_%']}%)")
        print(f"  Correções por campo:")
        for campo, count in cs["correcoes_por_campo"].items():
            print(f"    {campo:<22s}  {count:>4d} correções")
        print()

    # ── Resumo executivo ──
    if verbose:
        print("=" * 65)
        print("  RESUMO EXECUTIVO")
        print("=" * 65)
        print(f"  Extração OK:     {qj['taxa_extracao_ok']}%")
        print(f"  Redução naming:  {ag['taxa_reducao_%']}%  ({ag['total_registros_com_nome']} títulos → {ag['nomes_canonicos_unicos']} canônicos)")
        print(f"  Singletons:      {ag['taxa_singletons_%']}%  (nomes sem agrupamento — alvo de melhoria)")
        print(f"  Regex corrigiu:  {cs['taxa_correcao_%']}%  (indica margem de melhoria no prompt)")

        # Diagnósticos
        print()
        print("  Diagnósticos:")
        if qj["taxa_falha_total"] > 5:
            print(f"    ⚠️  Taxa de falha JSON ({qj['taxa_falha_total']}%) acima de 5% — revisar robustez do prompt")
        if ag["taxa_singletons_%"] > 70:
            print(f"    ⚠️  {ag['taxa_singletons_%']}% singletons — pipeline pouco efetivo em agrupar")
        if cs["taxa_correcao_%"] > 30:
            print(f"    ⚠️  Regex corrigiu {cs['taxa_correcao_%']}% dos registros — o prompt pode incorporar essas regras")
        for _, row in ca.iterrows():
            if row["alerta"]:
                print(f"    ⚠️  Campo '{row['campo']}' com apenas {row['taxa_preenchimento_%']}% de preenchimento")

        print()
        print("=" * 65)

    return relatorio


# ═══════════════════════════════════════════════════════════════════
#  EXECUÇÃO STANDALONE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/resultado_pipeline.csv"
    path = Path(csv_path)

    if not path.exists():
        print(f"Arquivo não encontrado: {csv_path}")
        print("Uso: python avaliacao_pipeline.py [caminho/para/resultado_pipeline.csv]")
        sys.exit(1)

    print(f"\nCarregando: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Registros: {len(df)}\n")

    relatorio = gerar_relatorio(df, verbose=True)