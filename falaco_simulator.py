!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
falaco_sim.py - Simulador simplificado dos "solitons de Falaco"
(par de vortices verticais de superficie gerado por um prato/remo puxado na agua).

O que o programa faz
--------------------
A partir de dados do "prato" (diametro, velocidade, curso ou forca/duracao) estima:
  * a circulacao inicial Gamma0 do vortice;
  * o crescimento do nucleo por difusao viscosa (Lamb-Oseen);
  * a velocidade de deslocamento U(t) e a distancia percorrida x(t) do par;
  * a profundidade da "covinha" na superficie h(t) e o perfil da superficie;
  * o tempo de vida visivel (covinha mais funda que um limiar h_min).

Modelo (resumo; as deducoes estao no PDF gerado pelo arquivo falaco_solitons.tex):
  Gamma0   = k * (1/2) * v0 * L                  (modelo "slug"; modo velocidade)
  Gamma0   = k * 2 * F * dt / (rho * pi * R^2)   (modo forca: impulso de meio anel)
  sigma^2  = a0^2 + 4 nu t                       (Lamb-Oseen)
  Gamma(t) = Gamma0 * exp(-lambda t)             (perdas extras: parametro empirico)
  U(t)     = Gamma/(4 pi R) * [ln(8R/sigma) - 0.558]   (anel viscoso, Saffman 1970)
           ou Gamma/(2 pi D)                           (par de vortices pontuais)
  h(t)     = Gamma^2 ln2 / (4 pi^2 g sigma^2)    (depressao central, Lamb-Oseen)

AVISO: e um modelo de ordem de grandeza, nao uma simulacao de Navier-Stokes.
O parametro de perdas (lambda) nao vem de primeiros principios: calibre-o com a
vida medida em video (opcao --vida-medida).

Dependencias: Python >= 3.9, numpy, matplotlib   (pip install -r requirements.txt)

Uso rapido:
  python falaco_sim.py                      # pergunta os valores (Enter = padrao)
  python falaco_sim.py --exemplo            # roda com os valores padrao
  python falaco_sim.py --diametro-cm 25 --velocidade 0.6 --curso-cm 40
  python falaco_sim.py --forca 8 --duracao 0.25 --diametro-cm 20
  python falaco_sim.py --vida-medida 45     # calibra a perda lambda para vida de 45 s
  python falaco_sim.py --gui                # janela com controles deslizantes
"""

import argparse
import math
import sys

import numpy as np

LN2 = math.log(2.0)
SAFFMAN = 0.558          # constante de Saffman (1970) para anel com nucleo de Lamb-Oseen
FORMACAO_GHARIB = 4.0    # "numero de formacao" L/D ~ 4 (Gharib, Rambod, Shariff 1998)


# ----------------------------------------------------------------------------
# Propriedades da agua
# ----------------------------------------------------------------------------
def propriedades_agua(temp_c):
    """Densidade (kg/m3) e viscosidade cinematica (m2/s) da agua doce a temp_c (C)."""
    T = float(temp_c)
    # Kell (1975), densidade
    rho = 1000.0 * (1.0 - (T + 288.9414) / (508929.2 * (T + 68.12963)) * (T - 3.9863) ** 2)
    # Viscosidade dinamica (Vogel), Pa.s
    mu = 2.414e-5 * 10.0 ** (247.8 / (T + 133.15))
    return rho, mu / rho


# ----------------------------------------------------------------------------
# Nucleo do calculo
# ----------------------------------------------------------------------------
def parametros_padrao():
    return dict(
        D=0.20,              # diametro/largura do prato (m)
        modo="velocidade",   # "velocidade" ou "forca"
        v0=0.50,             # velocidade do prato (m/s)
        L=0.30,              # curso do prato (m)
        F=5.0,               # forca aplicada (N)
        dt_golpe=0.20,       # duracao da forca (s)
        k_gamma=1.0,         # fator de eficiencia do enrolamento da vorticidade (0-1)
        a_rel=0.25,          # raio inicial do nucleo / R
        temp_c=20.0,         # temperatura da agua (C)
        nu=None,             # se None, calcula a partir da temperatura
        rho=None,
        g=9.81,
        h_min=0.5e-3,        # profundidade minima da covinha para ser "visivel" (m)
        lam=0.03,            # taxa de perda de circulacao (1/s), parametro empirico
        vida_medida=None,    # vida observada (s) para calibrar lam
        modelo="anel",       # "anel" ou "pontual"
        t_max=None,          # s; None = automatico
    )


def completar(p):
    p = dict(p)
    rho, nu = propriedades_agua(p["temp_c"])
    if p.get("rho") is None:
        p["rho"] = rho
    if p.get("nu") is None:
        p["nu"] = nu
    return p


def circulacao_inicial(p):
    """Gamma0 (m2/s), impulso do meio anel (kg m/s) e avisos."""
    R = p["D"] / 2.0
    rho = p["rho"]
    avisos = []
    if p["modo"] == "velocidade":
        gamma0 = p["k_gamma"] * 0.5 * p["v0"] * p["L"]
        impulso = 0.5 * rho * gamma0 * math.pi * R ** 2
        if p["L"] / p["D"] > FORMACAO_GHARIB:
            avisos.append(
                "L/D = %.1f > ~4: acima do 'numero de formacao' o anel tende a se "
                "desprender e a circulacao satura; Gamma0 pode estar superestimada."
                % (p["L"] / p["D"])
            )
    else:
        impulso = p["F"] * p["dt_golpe"]
        gamma0 = p["k_gamma"] * 2.0 * impulso / (rho * math.pi * R ** 2)
    return gamma0, impulso, avisos


def h_viscosa(t, gamma0, a0, nu, g):
    """Profundidade da covinha so com difusao viscosa (m)."""
    sigma2 = a0 ** 2 + 4.0 * nu * t
    return gamma0 ** 2 * LN2 / (4.0 * math.pi ** 2 * g * sigma2)


def vida_viscosa(gamma0, a0, nu, g, h_min):
    """Tempo (s) para h cair a h_min so por difusao viscosa."""
    alvo = gamma0 ** 2 * LN2 / (4.0 * math.pi ** 2 * g * h_min)
    return max(0.0, (alvo - a0 ** 2) / (4.0 * nu))


def vida_com_perdas(gamma0, a0, nu, g, h_min, lam):
    """Primeiro instante em que h(t) < h_min com perdas lam; inf se nunca (dentro de 1e7 s)."""
    def h(t):
        return h_viscosa(t, gamma0, a0, nu, g) * math.exp(-2.0 * lam * t)

    if h(0.0) <= h_min:
        return 0.0
    hi = 1.0
    while h(hi) > h_min:
        hi *= 2.0
        if hi > 1e7:
            return math.inf
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h(mid) > h_min:
            lo = mid
        else:
            hi = mid
    return hi


def perfil_superficie(gamma, sigma2, g, n=4000):
    """Perfil zeta(r) da superficie (m) para um vortice de Lamb-Oseen. Retorna r, zeta."""
    sigma = math.sqrt(sigma2)
    r_max = 8.0 * sigma
    r = np.linspace(1e-6, r_max, n)
    v = gamma / (2.0 * math.pi * r) * (1.0 - np.exp(-r ** 2 / sigma2))
    f = v ** 2 / r
    # integral de r ate r_max (da direita para a esquerda) + cauda analitica
    trap = 0.5 * (f[1:] + f[:-1]) * np.diff(r)
    acum = np.concatenate([np.cumsum(trap[::-1])[::-1], [0.0]])
    cauda = gamma ** 2 / (8.0 * math.pi ** 2 * r_max ** 2)
    zeta = -(acum + cauda) / g
    return r, zeta


def calcular(p):
    p = completar(p)
    R = p["D"] / 2.0
    nu, g = p["nu"], p["g"]
    gamma0, impulso, avisos = circulacao_inicial(p)
    a0 = p["a_rel"] * R

    h0 = h_viscosa(0.0, gamma0, a0, nu, g)
    t_visc = vida_viscosa(gamma0, a0, nu, g, p["h_min"])

    lam = p["lam"]
    lam_calibrada = False
    if p.get("vida_medida"):
        T = float(p["vida_medida"])
        hv = h_viscosa(T, gamma0, a0, nu, g)
        if hv > p["h_min"]:
            lam = math.log(hv / p["h_min"]) / (2.0 * T)
            lam_calibrada = True
        else:
            lam = 0.0
            avisos.append(
                "A covinha ja fica abaixo de h_min antes de %.0f s mesmo sem perdas; "
                "lambda = 0. Aumente a velocidade/curso ou reduza h_min." % T
            )
    t_vida = vida_com_perdas(gamma0, a0, nu, g, p["h_min"], lam)

    if p.get("t_max"):
        t_max = float(p["t_max"])
    elif math.isfinite(t_vida) and t_vida > 0:
        t_max = 1.6 * t_vida
    else:
        t_max = max(60.0, min(3600.0, 1.2 * t_visc))
    t = np.linspace(0.0, t_max, 2500)

    sigma2 = a0 ** 2 + 4.0 * nu * t
    sigma = np.sqrt(sigma2)
    gam = gamma0 * np.exp(-lam * t)
    h = gam ** 2 * LN2 / (4.0 * math.pi ** 2 * g * sigma2)

    if p["modelo"] == "anel":
        U = gam / (4.0 * math.pi * R) * (np.log(8.0 * R / sigma) - SAFFMAN)
    else:
        U = gam / (2.0 * math.pi * p["D"]) * np.ones_like(t)
    U = np.clip(U, 0.0, None)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (U[1:] + U[:-1]) * np.diff(t))])

    if h0 > 0.05 or h0 > R:
        avisos.append(
            "Covinha inicial de %.0f mm: muito funda para o modelo de superficie "
            "quase plana (pode haver nucleo de ar; Froude alto)." % (h0 * 1e3)
        )
    if a0 <= 0 or R <= 0:
        raise ValueError("Diametro e raio do nucleo devem ser positivos.")

    if math.isfinite(t_vida):
        i = int(np.searchsorted(t, min(t_vida, t[-1])))
        i = min(i, len(t) - 1)
        x_vida = float(x[i])
    else:
        x_vida = float(x[-1])

    return dict(
        p=p, R=R, a0=a0, gamma0=gamma0, impulso=impulso, h0=h0, lam=lam,
        lam_calibrada=lam_calibrada, t_visc=t_visc, t_vida=t_vida, x_vida=x_vida,
        U0=float(U[0]), Re=gamma0 / nu, t=t, sigma=sigma, sigma2=sigma2, gam=gam,
        h=h, U=U, x=x, avisos=avisos,
    )


def resumo_texto(res):
    p = res["p"]
    lin = []
    lin.append("=" * 62)
    lin.append("RESULTADOS (modelo simplificado - ordem de grandeza)")
    lin.append("=" * 62)
    lin.append("Prato: diametro %.1f cm | modo: %s" % (p["D"] * 100, p["modo"]))
    if p["modo"] == "velocidade":
        lin.append("  v0 = %.2f m/s, curso L = %.1f cm (L/D = %.2f)"
                   % (p["v0"], p["L"] * 100, p["L"] / p["D"]))
    else:
        lin.append("  F = %.2f N durante %.3f s (impulso %.3f N.s)"
                   % (p["F"], p["dt_golpe"], p["F"] * p["dt_golpe"]))
    lin.append("Agua: nu = %.3e m2/s, rho = %.1f kg/m3" % (p["nu"], p["rho"]))
    lin.append("-" * 62)
    lin.append("Circulacao inicial Gamma0      : %.4f m2/s" % res["gamma0"])
    lin.append("Reynolds de circulacao Gamma/nu: %.0f" % res["Re"])
    lin.append("Impulso do meio anel           : %.4f N.s" % res["impulso"])
    lin.append("Raio inicial do nucleo a0      : %.2f cm" % (res["a0"] * 100))
    lin.append("Velocidade inicial do par U0   : %.1f cm/s" % (res["U0"] * 100))
    lin.append("Profundidade inicial da covinha: %.2f mm" % (res["h0"] * 1e3))
    lin.append("Vida so com viscosidade        : %s" % formata_tempo(res["t_visc"]))
    lam_txt = "%.4f 1/s%s" % (res["lam"], " (calibrada)" if res["lam_calibrada"] else " (padrao/empirico)")
    lin.append("Perda extra de circulacao lambda: %s" % lam_txt)
    lin.append("Vida visivel (h > %.2f mm)     : %s" % (p["h_min"] * 1e3, formata_tempo(res["t_vida"])))
    lin.append("Distancia percorrida na vida   : %.2f m" % res["x_vida"])
    for a in res["avisos"]:
        lin.append("AVISO: " + a)
    lin.append("=" * 62)
    return "\n".join(lin)


def formata_tempo(t):
    if not math.isfinite(t):
        return "infinita (nunca atinge o limiar)"
    if t < 120:
        return "%.1f s" % t
    return "%.1f s (%.1f min)" % (t, t / 60.0)


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------
def criar_eixos(fig):
    ax = {}
    ax["U"] = fig.add_subplot(2, 2, 1)
    ax["x"] = ax["U"].twinx()
    ax["h"] = fig.add_subplot(2, 2, 2)
    ax["perfil"] = fig.add_subplot(2, 2, 3)
    ax["topo"] = fig.add_subplot(2, 2, 4)
    return ax


def desenhar(ax, res):
    import matplotlib.patches as mpatches

    for a in ax.values():
        a.clear()
    p = res["p"]
    t = res["t"]

    # 1) velocidade e distancia (eixo duplo: clear() desfaz o lado direito, entao refazemos)
    a1, a1b = ax["U"], ax["x"]
    a1b.yaxis.tick_right()
    a1b.yaxis.set_label_position("right")
    a1b.xaxis.set_visible(False)
    a1.plot(t, res["U"] * 100, color="tab:blue", label="velocidade U(t)")
    a1b.plot(t, res["x"], color="tab:red", label="distancia x(t)")
    a1.set_xlabel("tempo (s)")
    a1.set_ylabel("velocidade do par (cm/s)", color="tab:blue")
    a1b.set_ylabel("distancia percorrida (m)", color="tab:red")
    a1.set_title("Deslocamento do par")
    a1.grid(alpha=0.3)

    # 2) profundidade da covinha
    a2 = ax["h"]
    h_mm = res["h"] * 1e3
    a2.semilogy(t, np.clip(h_mm, 1e-6, None), color="tab:green", label="com perdas (lambda)")
    h_visc = np.array([h_viscosa(tt, res["gamma0"], res["a0"], p["nu"], p["g"]) for tt in t]) * 1e3
    a2.semilogy(t, h_visc, "--", color="gray", label="so viscosidade")
    a2.axhline(p["h_min"] * 1e3, color="k", lw=0.8, ls=":", label="limiar visivel")
    if math.isfinite(res["t_vida"]) and res["t_vida"] <= t[-1]:
        a2.axvline(res["t_vida"], color="tab:green", lw=0.8, ls=":")
    a2.set_xlabel("tempo (s)")
    a2.set_ylabel("profundidade da covinha (mm)")
    a2.set_title("Covinha na superficie")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, which="both")

    # 3) perfis da superficie
    a3 = ax["perfil"]
    ref = res["t_vida"] if math.isfinite(res["t_vida"]) and res["t_vida"] > 0 else t[-1]
    for frac, cor in zip([0.0, 0.25, 0.5, 1.0], ["tab:red", "tab:orange", "tab:olive", "tab:blue"]):
        tt = frac * ref
        s2 = res["a0"] ** 2 + 4.0 * p["nu"] * tt
        gm = res["gamma0"] * math.exp(-res["lam"] * tt)
        r, z = perfil_superficie(gm, s2, p["g"])
        r_plot = np.concatenate([-r[::-1], r])
        z_plot = np.concatenate([z[::-1], z])
        a3.plot(r_plot * 100, z_plot * 1e3, color=cor, label="t = %.0f s" % tt)
    a3.set_xlim(-0.5 * p["D"] * 100 * 1.5, 0.5 * p["D"] * 100 * 1.5)
    a3.set_xlabel("distancia ao eixo do vortice (cm)")
    a3.set_ylabel("nivel da superficie (mm)")
    a3.set_title("Perfil da covinha ao longo do tempo")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3)

    # 4) vista de cima
    a4 = ax["topo"]
    R = res["R"]
    xmax = max(res["x_vida"], 1e-3)
    t_fim = res["t_vida"] if math.isfinite(res["t_vida"]) else t[-1]
    sig_fim = math.sqrt(res["a0"] ** 2 + 4 * p["nu"] * t_fim)
    for sinal, cor in [(+1, "tab:red"), (-1, "tab:blue")]:
        y = sinal * R
        a4.plot([0, xmax], [y, y], color=cor, lw=1, alpha=0.6)
        # barras verticais = tamanho transversal do nucleo (escala sigma) no inicio e no fim
        a4.plot([0, 0], [y - res["a0"], y + res["a0"]], color=cor, lw=7, alpha=0.5,
                solid_capstyle="butt")
        a4.plot([xmax, xmax], [y - sig_fim, y + sig_fim], color=cor, lw=7, alpha=0.25,
                solid_capstyle="butt")
    a4.annotate("", xy=(xmax, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=1))
    a4.text(xmax / 2, 0.08 * R, "sentido do deslocamento", ha="center", fontsize=8)
    a4.set_title("Vista de cima: par de vortices (cor = sentido de giro)")
    a4.set_xlabel("distancia percorrida (m)")
    a4.set_ylabel("posicao transversal (m)")
    a4.set_xlim(-0.08 * xmax - 0.01, xmax * 1.08 + 0.01)
    a4.set_ylim(-1.7 * R, 1.7 * R)
    a4.grid(alpha=0.3)


def figura(res, titulo=True):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    ax = criar_eixos(fig)
    desenhar(ax, res)
    p = res["p"]
    if titulo:
        fig.suptitle(
            "Falaco: D=%.0f cm, Gamma0=%.3f m2/s, vida=%s, distancia=%.2f m"
            % (p["D"] * 100, res["gamma0"], formata_tempo(res["t_vida"]), res["x_vida"]),
            fontsize=11,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def modo_gui(p):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(12, 9))
    ax = criar_eixos(fig)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.34, hspace=0.50, wspace=0.40)

    if p["modo"] == "velocidade":
        specs = [("D", "Diametro (cm)", 5, 60, p["D"] * 100, 0.01),
                 ("v0", "Velocidade (m/s)", 0.05, 2.0, p["v0"], 1.0),
                 ("L", "Curso (cm)", 5, 100, p["L"] * 100, 0.01)]
    else:
        specs = [("D", "Diametro (cm)", 5, 60, p["D"] * 100, 0.01),
                 ("F", "Forca (N)", 0.5, 40, p["F"], 1.0),
                 ("dt_golpe", "Duracao (s)", 0.05, 1.0, p["dt_golpe"], 1.0)]
    specs += [("a_rel", "Nucleo a0/R", 0.05, 0.6, p["a_rel"], 1.0),
              ("lam", "Perda lambda (1/s)", 0.0, 0.2, p["lam"], 1.0)]

    sliders = {}
    for i, (chave, rot, vmin, vmax, vini, escala) in enumerate(specs):
        eixo = fig.add_axes([0.20, 0.21 - i * 0.04, 0.60, 0.025])
        sliders[chave] = (Slider(eixo, rot, vmin, vmax, valinit=vini), escala)

    def atualizar(_=None):
        q = dict(p)
        q["vida_medida"] = None
        for chave, (sl, escala) in sliders.items():
            q[chave] = sl.val * escala
        res = calcular(q)
        desenhar(ax, res)
        fig.suptitle("Gamma0=%.3f m2/s | U0=%.1f cm/s | covinha %.1f mm | vida %s | distancia %.2f m"
                     % (res["gamma0"], res["U0"] * 100, res["h0"] * 1e3,
                        formata_tempo(res["t_vida"]), res["x_vida"]), fontsize=10)
        fig.canvas.draw_idle()

    for sl, _ in sliders.values():
        sl.on_changed(atualizar)
    atualizar()
    return fig, sliders


# ----------------------------------------------------------------------------
# Entrada de dados
# ----------------------------------------------------------------------------
def pedir(texto, padrao, conv=float):
    while True:
        bruto = input("%s [%s]: " % (texto, padrao)).strip().replace(",", ".")
        if bruto == "":
            return padrao
        try:
            return conv(bruto)
        except ValueError:
            print("  valor invalido, tente de novo.")


def entrada_interativa(p):
    print("\nDigite os valores (Enter = usar o padrao entre colchetes).\n")
    p["D"] = pedir("Diametro/largura do prato (cm)", p["D"] * 100) / 100.0
    modo = pedir("Modo: 1 = velocidade+curso, 2 = forca+duracao", 1, int)
    if modo == 2:
        p["modo"] = "forca"
        p["F"] = pedir("Forca aplicada (N)", p["F"])
        p["dt_golpe"] = pedir("Duracao da forca (s)", p["dt_golpe"])
    else:
        p["modo"] = "velocidade"
        p["v0"] = pedir("Velocidade inicial do prato (m/s)", p["v0"])
        p["L"] = pedir("Curso/distancia puxada (cm)", p["L"] * 100) / 100.0
    p["a_rel"] = pedir("Raio do nucleo / raio do prato (0.1 a 0.5)", p["a_rel"])
    p["temp_c"] = pedir("Temperatura da agua (C)", p["temp_c"])
    p["lam"] = pedir("Perda extra de circulacao lambda (1/s)", p["lam"])
    vm = input("Vida medida do vortice em video (s) [Enter = nao informar]: ").strip().replace(",", ".")
    if vm:
        try:
            p["vida_medida"] = float(vm)
        except ValueError:
            print("  ignorado.")
    return p


def ler_argumentos(argv=None):
    ap = argparse.ArgumentParser(
        description="Simulador simplificado de solitons de Falaco (par de vortices de superficie).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--diametro-cm", type=float, help="diametro/largura do prato (cm)")
    ap.add_argument("--velocidade", type=float, help="velocidade do prato v0 (m/s)")
    ap.add_argument("--curso-cm", type=float, help="distancia puxada L (cm)")
    ap.add_argument("--forca", type=float, help="forca F (N) [usa modo forca]")
    ap.add_argument("--duracao", type=float, help="duracao da forca (s) [modo forca]")
    ap.add_argument("--nucleo", type=float, help="raio inicial do nucleo / R (padrao 0.25)")
    ap.add_argument("--eficiencia", type=float, help="fator k de enrolamento da vorticidade (padrao 1)")
    ap.add_argument("--temp-c", type=float, help="temperatura da agua (C, padrao 20)")
    ap.add_argument("--nu", type=float, help="viscosidade cinematica (m2/s); sobrepoe a temperatura")
    ap.add_argument("--perda", type=float, help="lambda: perda extra de circulacao (1/s)")
    ap.add_argument("--vida-medida", type=float, help="vida observada (s); calibra lambda")
    ap.add_argument("--h-min-mm", type=float, help="profundidade minima visivel da covinha (mm)")
    ap.add_argument("--modelo", choices=["anel", "pontual"], help="formula da velocidade do par")
    ap.add_argument("--tmax", type=float, help="tempo maximo dos graficos (s)")
    ap.add_argument("--saida", default="falaco_resultado.png", help="arquivo PNG de saida")
    ap.add_argument("--sem-janela", action="store_true", help="so salva o PNG, nao abre janela")
    ap.add_argument("--exemplo", action="store_true", help="usa os valores padrao, sem perguntas")
    ap.add_argument("--gui", action="store_true", help="abre janela com controles deslizantes")
    return ap.parse_args(argv)


def aplicar_argumentos(p, a):
    if a.diametro_cm is not None:
        p["D"] = a.diametro_cm / 100.0
    if a.velocidade is not None:
        p["v0"] = a.velocidade
    if a.curso_cm is not None:
        p["L"] = a.curso_cm / 100.0
    if a.forca is not None:
        p["F"] = a.forca
        p["modo"] = "forca"
    if a.duracao is not None:
        p["dt_golpe"] = a.duracao
    if a.nucleo is not None:
        p["a_rel"] = a.nucleo
    if a.eficiencia is not None:
        p["k_gamma"] = a.eficiencia
    if a.temp_c is not None:
        p["temp_c"] = a.temp_c
    if a.nu is not None:
        p["nu"] = a.nu
    if a.perda is not None:
        p["lam"] = a.perda
    if a.vida_medida is not None:
        p["vida_medida"] = a.vida_medida
    if a.h_min_mm is not None:
        p["h_min"] = a.h_min_mm * 1e-3
    if a.modelo is not None:
        p["modelo"] = a.modelo
    if a.tmax is not None:
        p["t_max"] = a.tmax
    return p


def principal(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a = ler_argumentos(argv)
    p = aplicar_argumentos(parametros_padrao(), a)

    passou_algum = any(getattr(a, k) is not None for k in
                       ["diametro_cm", "velocidade", "curso_cm", "forca", "duracao", "nucleo",
                        "eficiencia", "temp_c", "nu", "perda", "vida_medida", "h_min_mm",
                        "modelo", "tmax"])
    if not (a.exemplo or passou_algum or a.gui):
        p = entrada_interativa(p)

    import matplotlib
    if a.sem_janela:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.gui:
        p = completar(p)
        fig, _sliders = modo_gui(p)
        fig.savefig(a.saida, dpi=130)
        if not a.sem_janela:
            plt.show()
        return 0

    res = calcular(p)
    print(resumo_texto(res))
    fig = figura(res)
    fig.savefig(a.saida, dpi=150)
    print("Grafico salvo em:", a.saida)
    if not a.sem_janela:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(principal())
