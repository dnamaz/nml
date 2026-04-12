"""
Static opcode metadata for all 89 NML opcodes.

Each entry provides: category, description, operand schema, snippet template,
and all known aliases (classic, symbolic, verbose, extra).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpcodeInfo:
    canonical: str
    category: str
    description: str
    operand_schema: str          # human-readable: "Rd Rs1 Rs2"
    snippet: str                 # VSCode snippet with tabstops
    min_ops: int
    max_ops: int
    symbolic: str = ""
    verbose: str = ""
    aliases: tuple[str, ...] = ()
    constraints: str = ""        # shape/type constraints for data generation


OPCODES: dict[str, OpcodeInfo] = {}

def _reg(info: OpcodeInfo) -> None:
    OPCODES[info.canonical] = info


# ── Arithmetic (10) ─────────────────────────────────────────────────────────

_reg(OpcodeInfo("MMUL", "Arithmetic", "Matrix multiply: Rd = Rs1 @ Rs2",
    "Rd Rs1 Rs2", "MMUL ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="×", verbose="MATRIX_MULTIPLY",
    constraints="Rs1:[M,K] Rs2:[K,N] -> Rd:[M,N]. Inner dims must match. dtype:f32"))

_reg(OpcodeInfo("MADD", "Arithmetic", "Element-wise add: Rd = Rs1 + Rs2",
    "Rd Rs1 Rs2", "MADD ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="⊕", verbose="ADD", aliases=("ACCUMULATE",),
    constraints="Rs1 and Rs2 must broadcast (NumPy right-align rules). dtype:f32"))

_reg(OpcodeInfo("MSUB", "Arithmetic", "Element-wise subtract: Rd = Rs1 - Rs2",
    "Rd Rs1 Rs2", "MSUB ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="⊖", verbose="SUBTRACT",
    constraints="Rs1 and Rs2 must broadcast (NumPy right-align rules). dtype:f32"))

_reg(OpcodeInfo("EMUL", "Arithmetic", "Element-wise multiply: Rd = Rs1 * Rs2",
    "Rd Rs1 Rs2", "EMUL ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="⊗", verbose="ELEMENT_MULTIPLY", aliases=("SCALE",),
    constraints="Rs1 and Rs2 must broadcast (NumPy right-align rules). dtype:f32"))

_reg(OpcodeInfo("EDIV", "Arithmetic", "Element-wise divide: Rd = Rs1 / Rs2",
    "Rd Rs1 Rs2", "EDIV ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="⊘", verbose="ELEMENT_DIVIDE",
    constraints="Rs1 and Rs2 must broadcast. Rs2 must not contain zeros. dtype:f32"))

_reg(OpcodeInfo("SDOT", "Arithmetic", "Dot product: Rd = Rs1 · Rs2",
    "Rd Rs1 Rs2", "SDOT ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="·", verbose="DOT_PRODUCT", aliases=("DOT",),
    constraints="Rs1 and Rs2 must have same total element count. Output is scalar [1]. dtype:f32"))

_reg(OpcodeInfo("SCLR", "Arithmetic", "Scalar multiply: Rd = Rs * #imm (or Rd = Rs1 * Rs2)",
    "Rd Rs #imm", "SCLR ${1:Rd} ${2:Rs} ${3:#imm}", 2, 3,
    symbolic="∗", verbose="SCALE",
    constraints="Rs: any shape. #imm: float scalar. Output same shape as Rs. dtype:f32"))

_reg(OpcodeInfo("SDIV", "Arithmetic", "Scalar divide: Rd = Rs1 / Rs2|#imm",
    "Rd Rs1 Rs2|#imm", "SDIV ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="÷", verbose="DIVIDE", aliases=("SCALAR_DIVIDE",),
    constraints="Operates on first element (scalar). Rs2/imm must not be zero. dtype:f32"))

_reg(OpcodeInfo("SADD", "Arithmetic", "Scalar add: Rd = Rs + Rs2|#imm",
    "Rd Rs Rs2|#imm", "SADD ${1:Rd} ${2:Rs} ${3:Rs2}", 3, 3,
    symbolic="∔", verbose="SCALAR_ADD",
    constraints="Operates on first element (scalar). dtype:f32"))

_reg(OpcodeInfo("SSUB", "Arithmetic", "Scalar subtract: Rd = Rs - Rs2|#imm",
    "Rd Rs Rs2|#imm", "SSUB ${1:Rd} ${2:Rs} ${3:Rs2}", 3, 3,
    symbolic="∸", verbose="SCALAR_SUB",
    constraints="Operates on first element (scalar). dtype:f32"))

# ── Activation (4+1) ───────────────────────────────────────────────────────

_reg(OpcodeInfo("RELU", "Activation", "Rectified linear unit: Rd = max(0, Rs)",
    "Rd Rs", "RELU ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="⌐", verbose="RECTIFY",
    constraints="Rs: any shape. Output same shape. dtype:f32"))

_reg(OpcodeInfo("SIGM", "Activation", "Sigmoid: Rd = 1/(1+exp(-Rs))",
    "Rd Rs", "SIGM ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="σ", verbose="SIGMOID",
    constraints="Rs: any shape. Output same shape, values in (0,1). dtype:f32"))

_reg(OpcodeInfo("TANH", "Activation", "Hyperbolic tangent: Rd = tanh(Rs)",
    "Rd Rs", "TANH ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="τ", verbose="HYPERBOLIC_TANGENT",
    constraints="Rs: any shape. Output same shape, values in (-1,1). dtype:f32"))

_reg(OpcodeInfo("SOFT", "Activation", "Softmax: Rd = softmax(Rs)",
    "Rd Rs", "SOFT ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="Σ", verbose="SOFTMAX",
    constraints="Rs: [N,C] or [C]. Softmax over last dim. Output same shape, rows sum to 1. dtype:f32"))

# ── Memory (4) ─────────────────────────────────────────────────────────────

_reg(OpcodeInfo("LD", "Memory", "Load tensor from named memory slot",
    "Rd @name", "LD ${1:Rd} ${2:@name}", 2, 2,
    symbolic="↓", verbose="LOAD"))

_reg(OpcodeInfo("ST", "Memory", "Store tensor to named memory slot",
    "Rs @name", "ST ${1:Rs} ${2:@name}", 2, 2,
    symbolic="↑", verbose="STORE"))

_reg(OpcodeInfo("MOV", "Memory", "Copy register: Rd = Rs",
    "Rd Rs", "MOV ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="←", verbose="COPY"))

_reg(OpcodeInfo("ALLC", "Memory", "Allocate zero tensor with given shape",
    "Rd #[shape]", "ALLC ${1:Rd} ${2:#[1]}", 2, 2,
    symbolic="□", verbose="ALLOCATE"))

# ── Data Flow (4) ──────────────────────────────────────────────────────────

_reg(OpcodeInfo("RSHP", "Data Flow", "Reshape tensor to new dimensions",
    "Rd Rs [#shape]", "RSHP ${1:Rd} ${2:Rs} ${3:#[shape]}", 2, 3,
    symbolic="⊟", verbose="RESHAPE"))

_reg(OpcodeInfo("TRNS", "Data Flow", "Transpose tensor",
    "Rd [Rs]", "TRNS ${1:Rd} ${2:Rs}", 1, 2,
    symbolic="⊤", verbose="TRANSPOSE"))

_reg(OpcodeInfo("SPLT", "Data Flow", "Split tensor along dimension",
    "Rd Re Rs #dim", "SPLT ${1:Rd} ${2:Re} ${3:Rs} ${4:#dim}", 2, 4,
    symbolic="⊢", verbose="SPLIT"))

_reg(OpcodeInfo("MERG", "Data Flow", "Merge tensors along dimension",
    "Rd Rs1 Rs2 #dim", "MERG ${1:Rd} ${2:Rs1} ${3:Rs2} ${4:#dim}", 2, 4,
    symbolic="⊣", verbose="MERGE"))

# ── Comparison (3) ─────────────────────────────────────────────────────────

_reg(OpcodeInfo("CMP", "Comparison", "Compare two registers: sets flag RE = (Rs1[0] < Rs2[0])",
    "Rs1 Rs2", "CMP ${1:Rs1} ${2:Rs2}", 2, 2,
    symbolic="≶", verbose="COMPARE"))

_reg(OpcodeInfo("CMPI", "Comparison", "Compare register vs immediate: sets flag RE",
    "Rd Rs #imm", "CMPI ${1:Rd} ${2:Rs} ${3:#imm}", 2, 3,
    symbolic="≺", verbose="COMPARE_VALUE", aliases=("COMPARE_IMMEDIATE", "ϟ")))

_reg(OpcodeInfo("CMPF", "Comparison", "Feature comparison: Rd = (Rs[#feat] > #thresh)",
    "Rd Rs #feat #thresh", "CMPF ${1:Rd} ${2:Rs} ${3:#feat} ${4:#thresh}", 3, 4,
    symbolic="⋈", verbose="COMPARE_FEATURE"))

# ── Control Flow (5) ───────────────────────────────────────────────────────

_reg(OpcodeInfo("JMPT", "Control", "Jump if flag true: PC += offset + 1",
    "#offset", "JMPT ${1:#offset}", 1, 1,
    symbolic="↗", verbose="BRANCH_TRUE"))

_reg(OpcodeInfo("JMPF", "Control", "Jump if flag false: PC += offset + 1",
    "#offset", "JMPF ${1:#offset}", 1, 1,
    symbolic="↘", verbose="BRANCH_FALSE"))

_reg(OpcodeInfo("JUMP", "Control", "Unconditional jump: PC += offset + 1",
    "#offset", "JUMP ${1:#offset}", 1, 1,
    symbolic="→", verbose="BRANCH", aliases=("JMP",)))

_reg(OpcodeInfo("LOOP", "Control", "Begin counted loop over register or immediate",
    "Rs|#count", "LOOP ${1:Rs}", 0, 2,
    symbolic="↻", verbose="REPEAT", aliases=("BEGIN_LOOP",)))

_reg(OpcodeInfo("ENDP", "Control", "End of loop body",
    "", "ENDP", 0, 0,
    symbolic="↺", verbose="END_REPEAT", aliases=("END_LOOP",)))

# ── Subroutine (2) ─────────────────────────────────────────────────────────

_reg(OpcodeInfo("CALL", "Subroutine", "Call subroutine at offset (pushes return address)",
    "#offset", "CALL ${1:#offset}", 1, 2,
    symbolic="⇒", verbose="CALL", aliases=("CALL_SUB", "◎")))

_reg(OpcodeInfo("RET", "Subroutine", "Return from subroutine",
    "", "RET", 0, 0,
    symbolic="⇐", verbose="RETURN", aliases=("RETURN_SUB", "◉")))

# ── Tree (2) ───────────────────────────────────────────────────────────────

_reg(OpcodeInfo("LEAF", "Tree", "Load immediate constant into register",
    "Rd #value", "LEAF ${1:Rd} ${2:#value}", 2, 2,
    symbolic="∎", verbose="SET_VALUE", aliases=("SET_LEAF",),
    constraints="#value: float literal. Creates scalar [1] tensor. dtype:f32"))

_reg(OpcodeInfo("TACC", "Tree", "Accumulate: Rd = Rd + Rs (or Rd = Rs1 + Rs2)",
    "Rd Rs1 [Rs2]", "TACC ${1:Rd} ${2:Rs1} ${3:Rs2}", 2, 3,
    symbolic="∑", verbose="ACCUMULATE", aliases=("TREE_ACCUMULATE", "⨁")))

# ── System (3) ─────────────────────────────────────────────────────────────

_reg(OpcodeInfo("SYNC", "System", "Barrier synchronization (no-op in single-agent)",
    "", "SYNC", 0, 0,
    symbolic="⏸", verbose="BARRIER"))

_reg(OpcodeInfo("HALT", "System", "Terminate program execution",
    "", "HALT", 0, 0,
    symbolic="◼", verbose="STOP"))

_reg(OpcodeInfo("TRAP", "System", "Trigger runtime fault with optional error code",
    "[#code]", "TRAP ${1:#code}", 0, 1,
    symbolic="⚠", verbose="FAULT"))

# ── NML-V Vision (4) ───────────────────────────────────────────────────────

_reg(OpcodeInfo("CONV", "Vision", "2D convolution: Rd = Rs1 * Rs2 (kernel)",
    "Rd Rs Rkernel [#stride] [#pad]", "CONV ${1:Rd} ${2:Rs} ${3:Rkernel}", 3, 5,
    symbolic="⊛", verbose="CONVOLVE",
    constraints="2D: Rs:[H,W] Rkernel:[KH,KW]. 4D: Rs:[N,C,H,W] Rkernel:[Co,Ci,KH,KW] (Ci must match C). #stride default 1, #pad default 0. dtype:f32"))

_reg(OpcodeInfo("POOL", "Vision", "Max pooling with optional window size",
    "Rd Rs [#size] [#stride]", "POOL ${1:Rd} ${2:Rs}", 2, 4,
    symbolic="⊓", verbose="MAX_POOL",
    constraints="Rs:[H,W] or [N,C,H,W]. #size: pool window (default 2). Output spatial dims = floor((H-size)/stride)+1. dtype:f32"))

_reg(OpcodeInfo("UPSC", "Vision", "Upscale tensor by factor",
    "Rd Rs [#factor]", "UPSC ${1:Rd} ${2:Rs} ${3:#factor}", 2, 3,
    symbolic="⊔", verbose="UPSCALE",
    constraints="Rs:[H,W] or [N,C,H,W]. #factor: integer upscale multiplier. Output spatial dims *= factor. dtype:f32"))

_reg(OpcodeInfo("PADZ", "Vision", "Zero-pad tensor",
    "Rd Rs [#amount]", "PADZ ${1:Rd} ${2:Rs} ${3:#amount}", 2, 3,
    symbolic="⊡", verbose="ZERO_PAD",
    constraints="Rs:[H,W] or [N,C,H,W]. #amount: pixels of zero padding per side. Output spatial dims += 2*amount. dtype:f32"))

# ── NML-T Transformer (4) ──────────────────────────────────────────────────

_reg(OpcodeInfo("ATTN", "Transformer", "Multi-head attention: Rd = Attention(Q, K, [V])",
    "Rd Rq Rk [Rv]", "ATTN ${1:Rd} ${2:Rq} ${3:Rk} ${4:Rv}", 3, 4,
    symbolic="⊙", verbose="ATTENTION",
    constraints="Rq:[S,D] Rk:[S,D] (must match). Rv:[S,Dv] optional (defaults to Rk). Output:[S,Dv]. dtype:f32"))

_reg(OpcodeInfo("NORM", "Transformer", "Layer normalization",
    "Rd Rs [Rgamma] [Rbeta]", "NORM ${1:Rd} ${2:Rs}", 2, 4,
    symbolic="‖", verbose="LAYER_NORM",
    constraints="Rs: any shape. Rgamma/Rbeta: [last_dim] (scale/shift). Output same shape as Rs. dtype:f32"))

_reg(OpcodeInfo("EMBD", "Transformer", "Embedding lookup: Rd = Rtable[Rindex]",
    "Rd Rtable Rindex", "EMBD ${1:Rd} ${2:Rtable} ${3:Rindex}", 2, 3,
    symbolic="⊏", verbose="EMBED",
    constraints="Rtable:[V,D] (vocab_size, embed_dim). Rindex:[N] integer indices (0..V-1). Output:[N,D]. dtype:f32, Rindex values must be integers"))

_reg(OpcodeInfo("GELU", "Transformer", "Gaussian error linear unit activation",
    "Rd Rs", "GELU ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="ℊ", verbose="GELU",
    constraints="Rs: any shape. Output same shape. dtype:f32"))

# ── NML-R Reduction (4) ────────────────────────────────────────────────────

_reg(OpcodeInfo("RDUC", "Reduction", "Reduce tensor along dimension (sum/mean/max/min)",
    "Rd Rs [#dim] [#mode]", "RDUC ${1:Rd} ${2:Rs} ${3:#dim}", 2, 4,
    symbolic="⊥", verbose="REDUCE", aliases=("ϛ",),
    constraints="#dim: dimension to reduce (0=rows,1=cols). #mode: 0=sum,1=mean,2=max,3=min. Output loses reduced dim. dtype:f32"))

_reg(OpcodeInfo("WHER", "Reduction", "Conditional select: Rd = where(Rcond, Rs1, [Rs2])",
    "Rd Rcond Rs1 [Rs2]", "WHER ${1:Rd} ${2:Rcond} ${3:Rs1}", 3, 4,
    symbolic="⊻", verbose="WHERE",
    constraints="Rcond, Rs1, Rs2 must have same shape. Rcond values: 0 or 1. dtype:f32"))

_reg(OpcodeInfo("CLMP", "Reduction", "Clamp values: Rd = clamp(Rs, #min, #max)",
    "Rd Rs #min #max", "CLMP ${1:Rd} ${2:Rs} ${3:#min} ${4:#max}", 3, 4,
    symbolic="⊧", verbose="CLAMP",
    constraints="Rs: any shape. #min, #max: float. Output same shape, all values in [min,max]. dtype:f32"))

_reg(OpcodeInfo("CMPR", "Reduction", "Mask comparison: Rd = (Rs op #threshold)",
    "Rd Rs #op #threshold", "CMPR ${1:Rd} ${2:Rs} ${3:#op} ${4:#threshold}", 3, 4,
    symbolic="⊜", verbose="MASK_COMPARE",
    constraints="Rs: any shape. #op: 0=GT,1=LT,2=EQ,3=GE,4=LE. #threshold: float. Output same shape, values 0 or 1. dtype:f32"))

# ── NML-S Signal (2) ───────────────────────────────────────────────────────

_reg(OpcodeInfo("FFT", "Signal", "Fast Fourier Transform",
    "Rd Rs Rtwiddle", "FFT ${1:Rd} ${2:Rs} ${3:Rtwiddle}", 2, 3,
    symbolic="∿", verbose="FOURIER",
    constraints="Rs:[N] signal (N should be power of 2). Rtwiddle:[N] twiddle factors. Output:[N] complex magnitudes. dtype:f32"))

_reg(OpcodeInfo("FILT", "Signal", "Apply filter kernel to signal",
    "Rd Rs Rkernel [#mode]", "FILT ${1:Rd} ${2:Rs} ${3:Rkernel}", 3, 4,
    symbolic="⋐", verbose="FILTER",
    constraints="Rs:[N] signal. Rkernel:[K] filter coefficients (K <= N). Output:[N]. dtype:f32"))

# ── NML-M2M (13) ───────────────────────────────────────────────────────────

_reg(OpcodeInfo("META", "M2M", "Metadata annotation (no-op at runtime)",
    "@key \"value\"", "META ${1:@key} ${2:\"value\"}", 1, 2,
    symbolic="§", verbose="METADATA"))

_reg(OpcodeInfo("FRAG", "M2M", "Begin named fragment",
    "name", "FRAG ${1:name}", 1, 1,
    symbolic="◆", verbose="FRAGMENT"))

_reg(OpcodeInfo("ENDF", "M2M", "End fragment block",
    "", "ENDF", 0, 0,
    symbolic="◇", verbose="END_FRAGMENT"))

_reg(OpcodeInfo("LINK", "M2M", "Import external fragment by name",
    "@name", "LINK ${1:@name}", 1, 1,
    symbolic="⊚", verbose="IMPORT"))

_reg(OpcodeInfo("PTCH", "M2M", "Apply differential patch",
    "", "PTCH", 0, 0,
    symbolic="⊿", verbose="PATCH"))

_reg(OpcodeInfo("SIGN", "M2M", "Cryptographically sign program (agent=...)",
    "agent=...", "SIGN agent=${1:name}", 0, 0,
    symbolic="✦", verbose="SIGN_PROGRAM"))

_reg(OpcodeInfo("VRFY", "M2M", "Verify signature against public key",
    "@program @pubkey", "VRFY ${1:@program} ${2:@pubkey}", 2, 2,
    symbolic="✓", verbose="VERIFY_SIGNATURE"))

_reg(OpcodeInfo("VOTE", "M2M", "Consensus voting: Rd = vote(Rs, #strategy, [#threshold])",
    "Rd Rs #strategy [#threshold]", "VOTE ${1:Rd} ${2:Rs} ${3:#strategy}", 2, 4,
    symbolic="⚖", verbose="CONSENSUS"))

_reg(OpcodeInfo("PROJ", "M2M", "Linear projection: Rd = Rs1 @ Rs2",
    "Rd Rs1 Rs2", "PROJ ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="⟐", verbose="PROJECT"))

_reg(OpcodeInfo("DIST", "M2M", "Distance metric: Rd = dist(Rs1, Rs2, [#metric])",
    "Rd Rs1 Rs2 [#metric]", "DIST ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 4,
    symbolic="⟂", verbose="DISTANCE"))

_reg(OpcodeInfo("GATH", "M2M", "Gather: Rd = Rs1[Rs2]",
    "Rd Rs Rindex", "GATH ${1:Rd} ${2:Rs} ${3:Rindex}", 3, 3,
    symbolic="⊃", verbose="GATHER"))

_reg(OpcodeInfo("SCAT", "M2M", "Scatter: Rs -> Rd at Rindex (source first)",
    "Rs Rd Rindex", "SCAT ${1:Rs} ${2:Rd} ${3:Rindex}", 3, 3,
    symbolic="⊂", verbose="SCATTER", aliases=("SCTR",)))

# ── NML-TR Training (4) ────────────────────────────────────────────────────

_reg(OpcodeInfo("BKWD", "Training", "Backward pass: compute gradients",
    "Rgrad Ractivation Rloss [Rmask]",
    "BKWD ${1:Rgrad} ${2:Ractivation} ${3:Rloss}", 3, 4,
    symbolic="∇", verbose="BACKWARD"))

_reg(OpcodeInfo("WUPD", "Training", "Weight update: W -= lr * grad",
    "Rweights Rgrad Rlr [Rmomentum]",
    "WUPD ${1:Rweights} ${2:Rgrad} ${3:Rlr}", 3, 4,
    symbolic="⟳", verbose="WEIGHT_UPDATE"))

_reg(OpcodeInfo("LOSS", "Training", "Compute loss: Rd = loss(Rpred, Rlabel, [#type])",
    "Rd Rpred Rlabel [#type]",
    "LOSS ${1:Rd} ${2:Rpred} ${3:Rlabel}", 3, 4,
    symbolic="△", verbose="COMPUTE_LOSS",
    constraints="Rpred and Rlabel must have same shape [N,O]. #type: 0=MSE, 1=MAE, 2=cross-entropy. Output: scalar [1]. dtype:f32"))

_reg(OpcodeInfo("TNET", "Training", "Legacy training opcode (redirects to TRAIN). Prefer TRAIN+INFER for new code.",
    "Rconfig #epochs [#lr] [#optimizer]",
    "TNET ${1:Rconfig} ${2:#epochs}", 2, 9,
    symbolic="⥁", verbose="TRAIN_NETWORK",
    constraints="Rconfig:[L,3] where each row=[in_size,out_size,activation]. activation: 0=relu,1=sigmoid,2=tanh,3=softmax,4=none. Data: @input shape=[N,in_size], @labels shape=[N,final_out_size]. dtype:f32"))

# ── NML-TR Backward (11) ───────────────────────────────────────────────────

_reg(OpcodeInfo("RELUBK", "Training", "ReLU backward: gradient through ReLU",
    "Rd Rgrad Rinput", "RELUBK ${1:Rd} ${2:Rgrad} ${3:Rinput}", 3, 3,
    symbolic="⌐ˈ", verbose="RELU_BACKWARD", aliases=("RELU_BK",)))

_reg(OpcodeInfo("SIGMBK", "Training", "Sigmoid backward: gradient through sigmoid",
    "Rd Rgrad Rinput", "SIGMBK ${1:Rd} ${2:Rgrad} ${3:Rinput}", 3, 3,
    symbolic="σˈ", verbose="SIGMOID_BACKWARD", aliases=("SIGM_BK",)))

_reg(OpcodeInfo("TANHBK", "Training", "Tanh backward: gradient through tanh",
    "Rd Rgrad Rinput", "TANHBK ${1:Rd} ${2:Rgrad} ${3:Rinput}", 3, 3,
    symbolic="τˈ", verbose="TANH_BACKWARD", aliases=("TANH_BK",)))

_reg(OpcodeInfo("GELUBK", "Training", "GELU backward: gradient through GELU",
    "Rd Rgrad Rinput", "GELUBK ${1:Rd} ${2:Rgrad} ${3:Rinput}", 3, 3,
    symbolic="ℊˈ", verbose="GELU_BACKWARD", aliases=("GELU_BK",)))

_reg(OpcodeInfo("SOFTBK", "Training", "Softmax backward: gradient through softmax",
    "Rd Rgrad Rinput", "SOFTBK ${1:Rd} ${2:Rgrad} ${3:Rinput}", 3, 3,
    symbolic="Σˈ", verbose="SOFTMAX_BACKWARD", aliases=("SOFT_BK",)))

_reg(OpcodeInfo("MMULBK", "Training", "Matrix multiply backward: computes input and weight gradients",
    "Rd_dinput Rd_dweight Rgrad Rinput Rweight",
    "MMULBK ${1:Rd_dinput} ${2:Rd_dweight} ${3:Rgrad} ${4:Rinput} ${5:Rweight}", 5, 5,
    symbolic="×ˈ", verbose="MATMUL_BACKWARD", aliases=("MMUL_BK",)))

_reg(OpcodeInfo("CONVBK", "Training", "Convolution backward: computes input and kernel gradients",
    "Rd_dinput Rd_dkernel Rgrad Rinput Rkernel",
    "CONVBK ${1:Rd_dinput} ${2:Rd_dkernel} ${3:Rgrad} ${4:Rinput} ${5:Rkernel}", 5, 5,
    symbolic="⊛ˈ", verbose="CONV_BACKWARD", aliases=("CONV_BK",)))

_reg(OpcodeInfo("POOLBK", "Training", "Max pooling backward: routes gradient through argmax positions",
    "Rd Rgrad Rfwd_input [#pool_size] [#stride]",
    "POOLBK ${1:Rd} ${2:Rgrad} ${3:Rfwd_input}", 3, 5,
    symbolic="⊓ˈ", verbose="POOL_BACKWARD", aliases=("POOL_BK",)))

_reg(OpcodeInfo("NORMBK", "Training", "Layer norm backward: gradient through layer normalization",
    "Rd Rgrad Rinput", "NORMBK ${1:Rd} ${2:Rgrad} ${3:Rinput}", 3, 3,
    symbolic="‖ˈ", verbose="NORM_BACKWARD", aliases=("NORM_BK",)))

_reg(OpcodeInfo("ATTNBK", "Training", "Attention backward: computes Q, K, V gradients",
    "Rd_dq Rgrad Rq Rk Rv", "ATTNBK ${1:Rd_dq} ${2:Rgrad} ${3:Rq} ${4:Rk} ${5:Rv}", 5, 5,
    symbolic="⊙ˈ", verbose="ATTN_BACKWARD", aliases=("ATTN_BK",)))

_reg(OpcodeInfo("TNDEEP", "Training", "Legacy N-layer training opcode (redirects to TRAIN). Prefer TRAIN+INFER for new code.",
    "#epochs #lr #optimizer [@input_data] [@labels]",
    "TNDEEP ${1:#epochs} ${2:#lr} ${3:#optimizer}", 3, 5,
    symbolic="⥁ˈ", verbose="TRAIN_DEEP"))

# ── NML-TR v0.9 (4) ────────────────────────────────────────────────────────

_reg(OpcodeInfo("TLOG", "Training", "Set training log interval (print every N epochs)",
    "#n", "TLOG ${1:#n}", 0, 1,
    symbolic="⧖", verbose="TRAIN_LOG"))

_reg(OpcodeInfo("TRAIN", "Training", "Config-driven training: reads 6-element tensor [epochs, lr, optimizer, print_every, patience, min_delta]",
    "Rs [@input_data] [@labels]", "TRAIN ${1:Rs}", 1, 3,
    symbolic="⟴", verbose="TRAIN_CONFIG",
    constraints="Rs:[6] config tensor = [epochs, lr, optimizer(0=SGD,1=Adam,2=AdamW), print_every, patience, min_delta]. Requires prior TNET to define network. @input/@labels: named data slots. dtype:f32"))

_reg(OpcodeInfo("INFER", "Training", "Forward pass only (no weight update)",
    "Rd R_input", "INFER ${1:Rd} ${2:R_input}", 0, 2,
    symbolic="⟶", verbose="FORWARD_PASS",
    constraints="R_input must match network input shape from TNET config. Output Rd has shape [1,final_out_size]. Requires prior TNET+TRAIN. dtype:f32"))

_reg(OpcodeInfo("WDECAY", "Training", "Weight decay: Rd[i] *= (1 - lambda)",
    "Rd #lambda", "WDECAY ${1:Rd} ${2:#lambda}", 2, 2,
    symbolic="ω", verbose="WEIGHT_DECAY",
    constraints="Rd: weight tensor, any shape. #lambda: float decay rate (e.g. 0.0001). dtype:f32"))

# ── NML-TR Phase 3 (2) ────────────────────────────────────────────────────

_reg(OpcodeInfo("BN", "Training", "Batch normalization: Rd = norm(Rs, [Rgamma, [Rbeta]])",
    "Rd Rs [Rgamma] [Rbeta]", "BN ${1:Rd} ${2:Rs}", 2, 4,
    symbolic="⊞", verbose="BATCH_NORM",
    constraints="Rs:[N,F] batch data. Rgamma:[F] scale, Rbeta:[F] shift (optional, default 1/0). Output same shape as Rs. dtype:f32"))

_reg(OpcodeInfo("DROP", "Training", "Inverted dropout: Rd = dropout(Rs, #rate)",
    "Rd Rs [#rate]", "DROP ${1:Rd} ${2:Rs} ${3:#rate}", 2, 3,
    symbolic="≋", verbose="DROPOUT",
    constraints="Rs: any shape. #rate: float 0.0-1.0 (fraction to drop, 0.0=no dropout at inference). Output same shape. dtype:f32"))

# ── NML-G General (5) ──────────────────────────────────────────────────────

_reg(OpcodeInfo("SYS", "General", "System call with code",
    "Rd #code", "SYS ${1:Rd} ${2:#code}", 2, 2,
    symbolic="⚙", verbose="SYSTEM"))

_reg(OpcodeInfo("MOD", "General", "Modulo: Rd = Rs1 % Rs2",
    "Rd Rs1 Rs2", "MOD ${1:Rd} ${2:Rs1} ${3:Rs2}", 3, 3,
    symbolic="%", verbose="MODULO",
    constraints="Rs1, Rs2: scalar [1]. Values MUST be integers (casts to int internally). Rs2 must not be 0. Output: scalar [1]. dtype:i32"))

_reg(OpcodeInfo("ITOF", "General", "Integer to float conversion: Rd = float(Rs)",
    "Rd Rs", "ITOF ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="⊶", verbose="INT_TO_FLOAT",
    constraints="Rs: any shape with integer values. Output same shape. dtype:i32->f32"))

_reg(OpcodeInfo("FTOI", "General", "Float to integer conversion: Rd = int(Rs)",
    "Rd Rs", "FTOI ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="⊷", verbose="FLOAT_TO_INT",
    constraints="Rs: any shape with float values. Output same shape, truncated to int. dtype:f32->i32"))

_reg(OpcodeInfo("BNOT", "General", "Bitwise NOT: Rd = ~Rs",
    "Rd Rs", "BNOT ${1:Rd} ${2:Rs}", 2, 2,
    symbolic="¬", verbose="BITWISE_NOT",
    constraints="Rs: any shape. Operates on integer bit representation. dtype:i32"))


# ── Lookup helpers ──────────────────────────────────────────────────────────

_ALIAS_TO_CANONICAL: dict[str, str] = {}

for _info in OPCODES.values():
    _ALIAS_TO_CANONICAL[_info.canonical] = _info.canonical
    if _info.symbolic:
        _ALIAS_TO_CANONICAL[_info.symbolic] = _info.canonical
    if _info.verbose:
        _ALIAS_TO_CANONICAL[_info.verbose] = _info.canonical
    for _a in _info.aliases:
        _ALIAS_TO_CANONICAL[_a] = _info.canonical


def lookup(name: str) -> OpcodeInfo | None:
    """Look up an OpcodeInfo by any alias (classic, symbolic, verbose)."""
    canonical = _ALIAS_TO_CANONICAL.get(name) or _ALIAS_TO_CANONICAL.get(name.upper())
    if canonical:
        return OPCODES.get(canonical)
    return None


def get_constraints(opcode_names: list[str]) -> dict[str, str]:
    """Return {opcode: constraints} for the given opcodes that have constraints."""
    result = {}
    for name in opcode_names:
        info = lookup(name)
        if info and info.constraints:
            result[info.canonical] = info.constraints
    return result


def constraints_prompt(opcode_names: list[str]) -> str:
    """Build a concise constraints reference string for use in LLM prompts."""
    cs = get_constraints(opcode_names)
    if not cs:
        return ""
    lines = ["DATA CONSTRAINTS PER OPCODE:"]
    for op, c in cs.items():
        lines.append(f"  {op}: {c}")
    return "\n".join(lines)


# ── Register metadata ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegisterInfo:
    canonical: str
    index: int
    greek: str
    verbose: str
    purpose: str


REGISTERS: dict[str, RegisterInfo] = {}

_REG_DEFS: list[tuple[str, int, str, str, str]] = [
    ("R0",  0, "ι", "INPUT",       "General purpose tensor register"),
    ("R1",  1, "κ", "KERNEL",      "General purpose tensor register"),
    ("R2",  2, "λ", "LAYER",       "General purpose tensor register"),
    ("R3",  3, "μ", "MOMENTUM",    "General purpose tensor register"),
    ("R4",  4, "ν", "",            "General purpose tensor register"),
    ("R5",  5, "ξ", "",            "General purpose tensor register"),
    ("R6",  6, "ο", "",            "General purpose tensor register"),
    ("R7",  7, "π", "",            "General purpose tensor register"),
    ("R8",  8, "ρ", "",            "General purpose tensor register"),
    ("R9",  9, "ς", "",            "General purpose tensor register"),
    ("RA", 10, "α", "ACCUMULATOR", "Accumulator register"),
    ("RB", 11, "β", "GENERAL",     "General purpose"),
    ("RC", 12, "γ", "SCRATCH",     "Scratch / temporaries"),
    ("RD", 13, "δ", "COUNTER",     "Counter register"),
    ("RE", 14, "φ", "FLAG",        "Condition flag (set by CMP/CMPI/CMPF)"),
    ("RF", 15, "ψ", "STACK",       "Stack pointer"),
    ("RG", 16, "η", "GRAD1",       "Gradient register 1"),
    ("RH", 17, "θ", "GRAD2",       "Gradient register 2"),
    ("RI", 18, "ζ", "GRAD3",       "Gradient register 3"),
    ("RJ", 19, "ω", "LRATE",       "Learning rate register"),
    ("RK", 20, "χ", "",            "Extended GPR / training"),
    ("RL", 21, "υ", "",            "Extended GPR / training"),
    ("RM", 22, "ε", "",            "Extended GPR / training"),
    ("RN", 23, "",  "",            "Extended GPR"),
    ("RO", 24, "",  "",            "Extended GPR"),
    ("RP", 25, "",  "",            "Extended GPR"),
    ("RQ", 26, "",  "",            "Extended GPR"),
    ("RR", 27, "",  "",            "Extended GPR"),
    ("RS", 28, "",  "",            "Extended GPR"),
    ("RT", 29, "",  "",            "Extended GPR"),
    ("RU", 30, "",  "",            "Extended GPR"),
    ("RV", 31, "",  "",            "Extended GPR"),
]

_REG_ALIAS_TO_CANONICAL: dict[str, str] = {}

for _c, _i, _g, _v, _p in _REG_DEFS:
    info = RegisterInfo(_c, _i, _g, _v, _p)
    REGISTERS[_c] = info
    _REG_ALIAS_TO_CANONICAL[_c] = _c
    _REG_ALIAS_TO_CANONICAL[_c.lower()] = _c
    if _g:
        _REG_ALIAS_TO_CANONICAL[_g] = _c
    if _v:
        _REG_ALIAS_TO_CANONICAL[_v] = _c
        _REG_ALIAS_TO_CANONICAL[_v.lower()] = _c


def lookup_register(name: str) -> RegisterInfo | None:
    """Look up a RegisterInfo by any alias."""
    canonical = _REG_ALIAS_TO_CANONICAL.get(name) or _REG_ALIAS_TO_CANONICAL.get(name.upper())
    if canonical:
        return REGISTERS.get(canonical)
    return None
