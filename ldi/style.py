"""
Unified Figure Style for LDI-ES Project
=========================================
Import and call apply_style() before generating any figure.

Usage:
    from ldi.style import apply_style, COLORS, LINE_STYLES, FIGSIZES
    apply_style()
"""

import matplotlib.pyplot as plt

# ── Color Palette ─────────────────────────────────────────

COLORS = {
    # Model comparison (baseline, comparison, MC)
    'ES':     '#D62728',   # red (matplotlib 'tab:red')
    'VaR':    '#1F77B4',   # blue (matplotlib 'tab:blue')
    'Merton': '#7F7F7F',   # gray (matplotlib 'tab:gray')
    # Parameter variation (sensitivity ES-only figures, max 5 levels)
    'param_values': ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD'],
}

LINE_STYLES = {
    'ES':     {'color': COLORS['ES'],     'linestyle': '-',  'linewidth': 2.5},
    'VaR':    {'color': COLORS['VaR'],    'linestyle': '--', 'linewidth': 2.5},
    'Merton': {'color': COLORS['Merton'], 'linestyle': ':',  'linewidth': 2.0},
}

OPTION_DECOMP = {
    'put_k':   {'color': '#2CA02C', 'linestyle': '-',  'linewidth': 2.0},
    'neg_put': {'color': '#9467BD', 'linestyle': '--', 'linewidth': 2.0},
    'digital': {'color': '#17BECF', 'linestyle': '-.', 'linewidth': 2.0},
    'net':     {'color': '#1A1A1A', 'linestyle': '-',  'linewidth': 2.5},
}

# ── Figure Sizes ──────────────────────────────────────────

FIGSIZES = {
    'single': (8, 6),     # single-panel figures
    'triple': (16, 5),    # 1x3 subplots (allocation, fan charts)
    'quad':   (12, 10),   # 2x2 subplots (gamma compare, T compare)
}

# ── Output Settings ───────────────────────────────────────

DPI = 300

# ── Fan Chart & Histogram ─────────────────────────────────

FAN_ALPHA = {
    'outer':  0.12,   # 10-90% (or 5-95%) band
    'middle': 0.25,   # 25-75% band
    'inner':  0.40,   # 40-60% band
}

HIST_ALPHA = 0.5

# ── Grid Style ────────────────────────────────────────────

GRID = {'alpha': 0.3, 'linestyle': '-', 'linewidth': 0.5}

# ── Legend Style ──────────────────────────────────────────

LEGEND = {'loc': 'best', 'framealpha': 0.9, 'edgecolor': 'gray'}

# ── Reference Lines ───────────────────────────────────────

MERTON_LINE = {'color': COLORS['Merton'], 'linestyle': ':', 'linewidth': 1.5}
K_LINE = {'color': 'green', 'linestyle': '--', 'alpha': 0.5, 'linewidth': 1.0}
GAMBLING_REGION = {'color': COLORS['ES'], 'alpha': 0.15}


# ── Style Application ────────────────────────────────────

def apply_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        'font.size':            13,
        'axes.titlesize':       16,
        'axes.titleweight':     'bold',
        'axes.labelsize':       14,
        'legend.fontsize':      12,
        'xtick.labelsize':      12,
        'ytick.labelsize':      12,
        'figure.titlesize':     18,
        'figure.titleweight':   'bold',
        'font.family':          'serif',
        'mathtext.fontset':     'cm',
        'figure.dpi':           DPI,
        'savefig.dpi':          DPI,
        'savefig.bbox':         'tight',
    })


# ── Helper Functions ──────────────────────────────────────

def setup_grid(ax):
    """Apply standard grid to an axes."""
    ax.grid(True, **GRID)


def add_merton_hline(ax, value=1.0, label='Merton (A=1)'):
    """Add horizontal reference line at Merton level."""
    ax.axhline(value, **MERTON_LINE, label=label)


def add_k_vline(ax, k=1.0):
    """Add vertical reference line at target funding ratio k."""
    ax.axvline(k, **K_LINE)


def savefig(fig, path):
    """Save figure and close."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Paper-Specific Style (Journal Submission)
# ═══════════════════════════════════════════════════════════

# Warm palette for ES-only sensitivity figures (A1, B1, C1, D1, E1, C2, E2)
# Dark red → bright yellow (small parameter → large parameter)
WARM_PALETTE = ['#8B0000', '#D62728', '#FF6347', '#FFA500', '#FFD700']

# Paper line styles: Merton linewidth = 1.5 (vs 2.0 in exploratory)
PAPER_LINE_STYLES = {
    'ES':     {'color': COLORS['ES'],     'linestyle': '-',  'linewidth': 2.5},
    'VaR':    {'color': COLORS['VaR'],    'linestyle': '--', 'linewidth': 2.5},
    'Merton': {'color': COLORS['Merton'], 'linestyle': ':',  'linewidth': 1.5},
}

# A=1 reference line: gray dotted, linewidth 1.0
PAPER_REF_LINE = {'color': COLORS['Merton'], 'linestyle': ':', 'linewidth': 1.0}

# Paper grid: explicit light gray color
PAPER_GRID = {'alpha': 0.3, 'linewidth': 0.5, 'color': '#CCCCCC'}

# Paper gambling region shading (light red)
PAPER_GAMBLING = {'color': COLORS['ES'], 'alpha': 0.15}


def apply_paper_style():
    """Set matplotlib rcParams for paper submission figures.

    Same as apply_style() but with white background enforced.
    """
    plt.rcParams.update({
        'font.size':            13,
        'axes.titlesize':       16,
        'axes.titleweight':     'bold',
        'axes.labelsize':       14,
        'legend.fontsize':      12,
        'xtick.labelsize':      12,
        'ytick.labelsize':      12,
        'figure.titlesize':     18,
        'figure.titleweight':   'bold',
        'font.family':          'serif',
        'mathtext.fontset':     'cm',
        'figure.dpi':           DPI,
        'savefig.dpi':          DPI,
        'savefig.bbox':         'tight',
        'figure.facecolor':     'white',
        'axes.facecolor':       'white',
        'savefig.facecolor':    'white',
    })


def paper_grid(ax):
    """Apply paper-style grid to an axes."""
    ax.grid(True, **PAPER_GRID)


def paper_hline(ax, value=1.0, label='Merton (A=1)'):
    """Add A=1 reference horizontal line (paper style, lw=1.0)."""
    ax.axhline(value, **PAPER_REF_LINE, label=label)


def paper_savefig(fig, path):
    """Save figure with paper settings and close."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
