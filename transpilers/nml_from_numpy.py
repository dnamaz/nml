#!/usr/bin/env python3
"""
nml_from_numpy.py — Python/NumPy → NML transpiler

Converts a subset of Python+NumPy code into an NML program (.nml) and
optional data file (.nml.data).  Only the numeric/tensor subset of Python
is supported — strings, classes, dicts, exceptions, and I/O beyond print()
are rejected with clear error messages.

Supported constructs:
  - np.array([...])          → @tensor in .nml.data
  - np.zeros(shape)          → ALLC
  - np.dot / a @ b           → MMUL
  - a + b, a - b, a * b, a/b → MADD/MSUB/EMUL/EDIV or scalar variants
  - np.maximum(0, x)         → RELU
  - 1/(1+np.exp(-x))         → SIGM  (pattern-matched)
  - np.tanh(x)               → TANH
  - np.exp(x)                → SIGM/manual (via pattern)
  - np.sum / np.mean         → RDUC
  - np.reshape / x.reshape   → RSHP
  - x.T / np.transpose       → TRNS
  - for i in range(n):       → LOOP / ENDP
  - if cond: / else:         → CMPI + JMPF
  - print(...)               → SYS #0 (print numeric)
  - Function defs            → inlined (sigmoid, relu, etc.)

Usage:
    python3 transpilers/nml_from_numpy.py script.py
    python3 transpilers/nml_from_numpy.py script.py --output-nml out.nml --output-data out.nml.data
    python3 transpilers/nml_from_numpy.py script.py --dry-run
"""

import ast
import sys
import argparse
from pathlib import Path
from collections import OrderedDict


# ═══════════════════════════════════════════════════════════════════════════════
# Register file
# ═══════════════════════════════════════════════════════════════════════════════

NML_REGS = ['R0','R1','R2','R3','R4','R5','R6','R7','R8','R9',
             'RA','RB','RC','RD','RE','RF','RG','RH','RI','RJ',
             'RK','RL','RM','RN','RO','RP','RQ','RR','RS','RT','RU','RV']


class TranspileError(Exception):
    """Raised when a Python construct cannot be transpiled to NML."""
    def __init__(self, message, node=None):
        self.node = node
        lineno = getattr(node, 'lineno', '?')
        super().__init__(f"line {lineno}: {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# Core transpiler
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyToNML:
    def __init__(self, source: str, filename: str = "<input>", verbose: bool = False):
        self.source = source
        self.filename = filename
        self.verbose = verbose

        # State
        self.instructions: list[str] = []
        self.data_tensors: OrderedDict = OrderedDict()  # name -> (shape, values)
        self.var_map: dict[str, str] = {}      # python var name -> NML register or @slot
        self.func_map: dict[str, ast.FunctionDef] = {}  # user-defined functions
        self.reg_alloc: int = 0                # next register index
        self.data_counter: int = 0             # for auto-naming data slots
        self.warnings: list[str] = []
        self.loop_depth: int = 0

    def _alloc_reg(self) -> str:
        if self.reg_alloc >= 29:
            raise TranspileError("Register overflow: program too complex (>29 variables). "
                                 "Consider simplifying or splitting into multiple programs.")
        reg = NML_REGS[self.reg_alloc]
        self.reg_alloc += 1
        return reg

    def _emit(self, instruction: str):
        indent = "  " * self.loop_depth
        self.instructions.append(f"{indent}{instruction}")

    def _emit_comment(self, text: str):
        indent = "  " * self.loop_depth
        self.instructions.append(f"{indent}; {text}")

    def _safe_name(self, name: str) -> str:
        return name.replace(' ', '_').replace('-', '_')[:32]

    def _add_data(self, name: str, values: list, shape: tuple) -> str:
        """Add a tensor to the data file. Returns the @slot name."""
        slot = self._safe_name(name)
        self.data_tensors[slot] = (shape, values)
        return f"@{slot}"

    # ─── Pattern detection helpers ────────────────────────────────────────────

    def _is_np_call(self, node: ast.expr, func_name: str) -> bool:
        """Check if node is np.func_name(...)"""
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            return f.value.id in ('np', 'numpy') and f.attr == func_name
        return False

    def _is_np_attr(self, node: ast.expr, attr_name: str) -> bool:
        """Check if node is np.attr_name"""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id in ('np', 'numpy') and node.attr == attr_name
        return False

    def _is_sigmoid_pattern(self, node: ast.expr) -> ast.expr | None:
        """Detect 1 / (1 + np.exp(-x)) and return x, or None."""
        # Pattern: BinOp(1, Div, BinOp(1, Add, Call(np.exp, UnaryOp(USub, x))))
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            return None
        if not self._is_const(node.left, 1):
            return None
        denom = node.right
        # denom should be (1 + np.exp(-x))
        if not isinstance(denom, ast.BinOp) or not isinstance(denom.op, ast.Add):
            return None
        if not self._is_const(denom.left, 1):
            return None
        exp_call = denom.right
        if not self._is_np_call(exp_call, 'exp'):
            return None
        if len(exp_call.args) != 1:
            return None
        neg_x = exp_call.args[0]
        if isinstance(neg_x, ast.UnaryOp) and isinstance(neg_x.op, ast.USub):
            return neg_x.operand
        return None

    def _is_relu_pattern(self, node: ast.expr) -> ast.expr | None:
        """Detect np.maximum(0, x) and return x."""
        if self._is_np_call(node, 'maximum') and len(node.args) == 2:
            if self._is_const(node.args[0], 0):
                return node.args[1]
            if self._is_const(node.args[1], 0):
                return node.args[0]
        return None

    def _is_const(self, node: ast.expr, value=None) -> bool:
        """Check if node is a numeric constant, optionally matching a specific value."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return value is None or node.value == value
        return False

    def _get_const(self, node: ast.expr) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    def _is_scalar(self, node: ast.expr) -> bool:
        """Check if node is a plain numeric constant."""
        return self._is_const(node)

    # ─── Expression compilation ───────────────────────────────────────────────

    def _compile_expr(self, node: ast.expr) -> str:
        """Compile an expression, return the NML register holding the result."""

        # ── Numeric literal ──
        if self._is_const(node):
            reg = self._alloc_reg()
            val = self._get_const(node)
            self._emit(f"LEAF  {reg} #{val}")
            return reg

        # ── Variable reference ──
        if isinstance(node, ast.Name):
            if node.id in self.var_map:
                ref = self.var_map[node.id]
                if ref.startswith('@'):
                    reg = self._alloc_reg()
                    self._emit(f"LD    {reg} {ref}")
                    return reg
                return ref
            raise TranspileError(f"Undefined variable: {node.id}", node)

        # ── Unary negation ──
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._compile_expr(node.operand)
            reg = self._alloc_reg()
            self._emit(f"SCLR  {reg} {inner} #-1.0")
            return reg

        # ── Sigmoid pattern: 1/(1+np.exp(-x)) ──
        sigmoid_arg = self._is_sigmoid_pattern(node)
        if sigmoid_arg is not None:
            src = self._compile_expr(sigmoid_arg)
            reg = self._alloc_reg()
            self._emit(f"SIGM  {reg} {src}")
            return reg

        # ── ReLU pattern: np.maximum(0, x) ──
        relu_arg = self._is_relu_pattern(node)
        if relu_arg is not None:
            src = self._compile_expr(relu_arg)
            reg = self._alloc_reg()
            self._emit(f"RELU  {reg} {src}")
            return reg

        # ── np.array([...]) ──
        if self._is_np_call(node, 'array') and len(node.args) >= 1:
            return self._compile_np_array(node)

        # ── np.zeros / np.ones ──
        if self._is_np_call(node, 'zeros'):
            return self._compile_np_alloc(node, fill=0.0)
        if self._is_np_call(node, 'ones'):
            return self._compile_np_alloc(node, fill=1.0)

        # ── np.dot(a, b) ──
        if self._is_np_call(node, 'dot') and len(node.args) == 2:
            a = self._compile_expr(node.args[0])
            b = self._compile_expr(node.args[1])
            reg = self._alloc_reg()
            self._emit(f"MMUL  {reg} {a} {b}")
            return reg

        # ── np.exp(x) ──
        if self._is_np_call(node, 'exp'):
            # NML has no standalone EXP opcode — warn and use SIGM workaround
            self.warnings.append("np.exp() has no direct NML opcode. "
                                 "Consider using sigmoid: 1/(1+np.exp(-x)) → SIGM")
            src = self._compile_expr(node.args[0])
            # Approximate: we can't do raw exp, but if this is part of a larger
            # expression it might get pattern-matched. Emit a placeholder comment.
            self._emit_comment(f"WARNING: raw np.exp() — no NML opcode, register {src} unchanged")
            return src

        # ── np.tanh(x) ──
        if self._is_np_call(node, 'tanh'):
            src = self._compile_expr(node.args[0])
            reg = self._alloc_reg()
            self._emit(f"TANH  {reg} {src}")
            return reg

        # ── np.sum(x) / np.mean(x) ──
        if self._is_np_call(node, 'sum'):
            src = self._compile_expr(node.args[0])
            reg = self._alloc_reg()
            self._emit(f"RDUC  {reg} {src} #0")  # mode 0 = sum
            return reg
        if self._is_np_call(node, 'mean'):
            src = self._compile_expr(node.args[0])
            reg = self._alloc_reg()
            self._emit(f"RDUC  {reg} {src} #1")  # mode 1 = mean
            return reg

        # ── np.reshape(x, shape) / x.reshape(shape) ──
        if self._is_np_call(node, 'reshape') and len(node.args) >= 2:
            src = self._compile_expr(node.args[0])
            shape = self._extract_shape(node.args[1])
            reg = self._alloc_reg()
            self._emit(f"RSHP  {reg} {src} #[{",".join(str(s) for s in shape)}]")
            return reg
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'reshape':
                src = self._compile_expr(node.func.value)
                shape = self._extract_shape_from_args(node.args)
                reg = self._alloc_reg()
                self._emit(f"RSHP  {reg} {src} #[{",".join(str(s) for s in shape)}]")
                return reg

        # ── x.T / np.transpose(x) ──
        if isinstance(node, ast.Attribute) and node.attr == 'T':
            src = self._compile_expr(node.value)
            reg = self._alloc_reg()
            self._emit(f"TRNS  {reg} {src}")
            return reg
        if self._is_np_call(node, 'transpose'):
            src = self._compile_expr(node.args[0])
            reg = self._alloc_reg()
            self._emit(f"TRNS  {reg} {src}")
            return reg

        # ── np.sqrt(x) ──
        if self._is_np_call(node, 'sqrt'):
            src = self._compile_expr(node.args[0])
            reg = self._alloc_reg()
            self._emit(f"SCLR  {reg} {src} #0.5")  # approximate: x^0.5 via scale
            self.warnings.append("np.sqrt() approximated via SCLR #0.5 (element-wise multiply, not true sqrt)")
            return reg

        # ── np.abs(x) / np.clip(x, min, max) ──
        if self._is_np_call(node, 'clip') and len(node.args) >= 3:
            src = self._compile_expr(node.args[0])
            mn = self._get_const(node.args[1])
            mx = self._get_const(node.args[2])
            if mn is not None and mx is not None:
                reg = self._alloc_reg()
                self._emit(f"CLMP  {reg} {src} #{mn} #{mx}")
                return reg

        # ── np.softmax — scipy.special.softmax or manual ──
        if self._is_np_call(node, 'softmax'):
            src = self._compile_expr(node.args[0])
            reg = self._alloc_reg()
            self._emit(f"SOFT  {reg} {src}")
            return reg

        # ── Binary ops: +, -, *, /, @, ** ──
        if isinstance(node, ast.BinOp):
            return self._compile_binop(node)

        # ── Comparison: a >= b, a < b, etc. ──
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return self._compile_compare(node)

        # ── Subscript: a[i] ──
        if isinstance(node, ast.Subscript):
            # For simple indexing, compile the base and note the limitation
            self.warnings.append(f"line {node.lineno}: Subscript indexing has limited NML support")
            return self._compile_expr(node.value)

        # ── Function call (user-defined) ──
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in self.func_map:
                return self._compile_func_call(fname, node.args)
            # Built-in print handled at statement level
            raise TranspileError(f"Unknown function: {fname}()", node)

        raise TranspileError(
            f"Unsupported expression: {ast.dump(node)}", node)

    def _compile_np_array(self, node: ast.Call) -> str:
        """Compile np.array([...]) into a data slot + LD."""
        arg = node.args[0]
        values = self._extract_list_values(arg)
        if values is None:
            raise TranspileError("np.array() argument must be a literal list", node)

        # Determine shape
        if isinstance(values[0], list):
            # 2D
            rows = len(values)
            cols = len(values[0])
            flat = [v for row in values for v in row]
            shape = (rows, cols)
        else:
            flat = values
            shape = (len(flat), 1)

        self.data_counter += 1
        name = f"array_{self.data_counter}"
        slot = self._add_data(name, flat, shape)

        reg = self._alloc_reg()
        self._emit(f"LD    {reg} {slot}")
        return reg

    def _compile_np_alloc(self, node: ast.Call, fill: float) -> str:
        """Compile np.zeros/np.ones into ALLC + optional fill."""
        shape = self._extract_shape(node.args[0]) if node.args else (1,)
        reg = self._alloc_reg()
        shape_str = ",".join(str(s) for s in shape)
        self._emit(f"ALLC  {reg} #[{shape_str}]")
        if fill != 0.0:
            self._emit(f"SADD  {reg} {reg} #{fill}")
        return reg

    def _compile_binop(self, node: ast.BinOp) -> str:
        """Compile a binary operation."""
        op = node.op

        # MatMul: a @ b
        if isinstance(op, ast.MatMult):
            a = self._compile_expr(node.left)
            b = self._compile_expr(node.right)
            reg = self._alloc_reg()
            self._emit(f"MMUL  {reg} {a} {b}")
            return reg

        # Check for scalar right operand → use scalar ops
        right_const = self._get_const(node.right)
        left_const = self._get_const(node.left)

        if isinstance(op, ast.Add):
            if right_const is not None:
                a = self._compile_expr(node.left)
                reg = self._alloc_reg()
                self._emit(f"SADD  {reg} {a} #{right_const}")
                return reg
            if left_const is not None:
                b = self._compile_expr(node.right)
                reg = self._alloc_reg()
                self._emit(f"SADD  {reg} {b} #{left_const}")
                return reg
            a = self._compile_expr(node.left)
            b = self._compile_expr(node.right)
            reg = self._alloc_reg()
            self._emit(f"MADD  {reg} {a} {b}")
            return reg

        if isinstance(op, ast.Sub):
            if right_const is not None:
                a = self._compile_expr(node.left)
                reg = self._alloc_reg()
                self._emit(f"SSUB  {reg} {a} #{right_const}")
                return reg
            a = self._compile_expr(node.left)
            b = self._compile_expr(node.right)
            reg = self._alloc_reg()
            self._emit(f"MSUB  {reg} {a} {b}")
            return reg

        if isinstance(op, ast.Mult):
            if right_const is not None:
                a = self._compile_expr(node.left)
                reg = self._alloc_reg()
                self._emit(f"SCLR  {reg} {a} #{right_const}")
                return reg
            if left_const is not None:
                b = self._compile_expr(node.right)
                reg = self._alloc_reg()
                self._emit(f"SCLR  {reg} {b} #{left_const}")
                return reg
            a = self._compile_expr(node.left)
            b = self._compile_expr(node.right)
            reg = self._alloc_reg()
            self._emit(f"EMUL  {reg} {a} {b}")
            return reg

        if isinstance(op, ast.Div):
            if right_const is not None:
                a = self._compile_expr(node.left)
                reg = self._alloc_reg()
                self._emit(f"SDIV  {reg} {a} #{right_const}")
                return reg
            a = self._compile_expr(node.left)
            b = self._compile_expr(node.right)
            reg = self._alloc_reg()
            self._emit(f"EDIV  {reg} {a} {b}")
            return reg

        if isinstance(op, ast.FloorDiv):
            self.warnings.append(f"line {node.lineno}: Floor division (//) approximated as regular division")
            if right_const is not None:
                a = self._compile_expr(node.left)
                reg = self._alloc_reg()
                self._emit(f"SDIV  {reg} {a} #{right_const}")
                return reg
            a = self._compile_expr(node.left)
            b = self._compile_expr(node.right)
            reg = self._alloc_reg()
            self._emit(f"EDIV  {reg} {a} {b}")
            return reg

        if isinstance(op, ast.Pow):
            # x ** 2 → EMUL x x
            if right_const == 2:
                a = self._compile_expr(node.left)
                reg = self._alloc_reg()
                self._emit(f"EMUL  {reg} {a} {a}")
                return reg
            self.warnings.append(f"line {node.lineno}: Power operator (**) only supports **2")

        raise TranspileError(f"Unsupported operator: {type(op).__name__}", node)

    def _compile_compare(self, node: ast.Compare) -> str:
        """Compile a comparison into CMPI + RE flag."""
        left = self._compile_expr(node.left)
        right = self._compile_expr(node.comparators[0])
        op = node.ops[0]

        # NML CMP sets RE (flag register) based on comparison
        # CMP RE Ra Rb — sets RE based on Ra < Rb
        self._emit(f"CMP   {left} {right}")

        # The result is in RE (R14 / condition flag)
        # For >= we need to invert
        if isinstance(op, (ast.GtE, ast.Gt)):
            self.warnings.append(f"line {node.lineno}: Comparison sets RE flag — "
                                 f"use with JMPT/JMPF for control flow")
        return "RE"

    def _compile_func_call(self, name: str, args: list) -> str:
        """Inline a user-defined function call."""
        func_def = self.func_map[name]

        # Save current var_map, bind arguments
        saved = dict(self.var_map)
        for param, arg_node in zip(func_def.args.args, args):
            param_name = param.arg
            arg_reg = self._compile_expr(arg_node)
            self.var_map[param_name] = arg_reg

        # Compile function body — last expression is the return value
        result_reg = None
        for stmt in func_def.body:
            # Skip docstrings
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
               and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Return) and stmt.value:
                result_reg = self._compile_expr(stmt.value)
            elif isinstance(stmt, ast.Assign):
                self._compile_assign(stmt)
            elif isinstance(stmt, ast.Expr):
                result_reg = self._compile_expr(stmt.value)

        # Restore var_map
        self.var_map = saved
        return result_reg or "R0"

    # ─── Statement compilation ────────────────────────────────────────────────

    def _compile_assign(self, node: ast.Assign):
        """Compile variable assignment."""
        if len(node.targets) != 1:
            raise TranspileError("Multiple assignment targets not supported", node)
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            # Tuple unpacking, subscript assignment, etc.
            raise TranspileError(f"Unsupported assignment target: {type(target).__name__}", node)

        var_name = target.id
        reg = self._compile_expr(node.value)

        # Inside loops: if the variable already has a register, MOV the result
        # back into its existing register so the next iteration sees the update.
        if self.loop_depth > 0 and var_name in self.var_map:
            existing = self.var_map[var_name]
            if existing.startswith('R') and existing != reg:
                self._emit(f"MOV   {existing} {reg}")
                # Don't update var_map — keep using the original register
                if self.verbose:
                    self._emit_comment(f"{var_name} updated in-place → {existing}")
                return

        self.var_map[var_name] = reg

        if self.verbose:
            self._emit_comment(f"{var_name} = {reg}")

    def _compile_for(self, node: ast.For):
        """Compile for i in range(n): → LOOP/ENDP."""
        # Only support: for var in range(N)
        if not isinstance(node.iter, ast.Call):
            raise TranspileError("Only 'for x in range(N)' loops are supported", node)
        func = node.iter.func
        if not (isinstance(func, ast.Name) and func.id == 'range'):
            raise TranspileError("Only 'for x in range(N)' loops are supported", node)

        args = node.iter.args
        if len(args) == 1:
            n_const = self._get_const(args[0])
            if n_const is not None:
                self._emit(f"LOOP  #{int(n_const)}")
            else:
                # Variable loop count — compile into a register
                n_reg = self._compile_expr(args[0])
                self._emit(f"LOOP  {n_reg}")
        else:
            raise TranspileError("range() with start/step not supported — use range(N)", node)

        # Loop variable (if referenced, map to RD = loop counter)
        if isinstance(node.target, ast.Name):
            self.var_map[node.target.id] = "RD"

        self.loop_depth += 1
        for stmt in node.body:
            self._compile_statement(stmt)
        self.loop_depth -= 1

        self._emit("ENDP")

        if node.orelse:
            self.warnings.append(f"line {node.lineno}: 'else' clause on for loop ignored")

    def _compile_if(self, node: ast.If):
        """Compile if/else using CMPI + JMPF."""
        # Compile the condition
        self._compile_expr(node.test)

        # Count instructions in the 'then' branch to know jump offset
        saved_instructions = list(self.instructions)
        saved_len = len(self.instructions)

        # Compile then-branch to measure its size
        then_instructions = []
        for stmt in node.body:
            self._compile_statement(stmt)
        then_count = len(self.instructions) - saved_len

        if node.orelse:
            then_count += 1  # account for the JUMP at end of then-branch

        # Reset and emit properly with jump
        self.instructions = saved_instructions
        self._emit(f"JMPF  #{then_count}")

        for stmt in node.body:
            self._compile_statement(stmt)

        if node.orelse:
            # Measure else branch
            else_start = len(self.instructions)
            saved2 = list(self.instructions)
            for stmt in node.orelse:
                self._compile_statement(stmt)
            else_count = len(self.instructions) - else_start

            self.instructions = saved2
            self._emit(f"JUMP  #{else_count}")
            for stmt in node.orelse:
                self._compile_statement(stmt)

    def _compile_print(self, node: ast.Call):
        """Compile print() → SYS #0 for numeric values."""
        for arg in node.args:
            # Skip string literals and f-strings
            if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self._emit_comment(f'print: "{arg.value}"')
                    continue
                if isinstance(arg, ast.JoinedStr):
                    # f-string — try to extract numeric parts
                    for val in arg.values:
                        if isinstance(val, ast.FormattedValue):
                            reg = self._compile_expr(val.value)
                            self._emit(f"SYS   {reg} #0")
                    continue
            reg = self._compile_expr(arg)
            self._emit(f"SYS   {reg} #0")

    def _compile_statement(self, node: ast.stmt):
        """Compile a single statement."""
        if isinstance(node, ast.Assign):
            self._compile_assign(node)

        elif isinstance(node, ast.AugAssign):
            # x += val, x -= val, etc.
            if not isinstance(node.target, ast.Name):
                raise TranspileError("Augmented assignment only for simple variables", node)
            var = node.target.id
            if var not in self.var_map:
                raise TranspileError(f"Undefined variable: {var}", node)

            # Build a synthetic BinOp and compile it
            fake_binop = ast.BinOp(
                left=ast.Name(id=var, ctx=ast.Load()),
                op=node.op,
                right=node.value
            )
            ast.copy_location(fake_binop, node)
            ast.copy_location(fake_binop.left, node)
            reg = self._compile_expr(fake_binop)
            self.var_map[var] = reg

        elif isinstance(node, ast.For):
            self._compile_for(node)

        elif isinstance(node, ast.If):
            self._compile_if(node)

        elif isinstance(node, ast.Expr):
            # Skip docstrings and standalone string literals
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return
            # Expression statement — usually a function call
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == 'print':
                    self._compile_print(node.value)
                    return
            self._compile_expr(node.value)

        elif isinstance(node, ast.FunctionDef):
            # Store for later inlining
            self.func_map[node.name] = node
            if self.verbose:
                self._emit_comment(f"Function defined: {node.name}()")

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            # Skip import statements
            pass

        elif isinstance(node, ast.Return):
            if node.value:
                reg = self._compile_expr(node.value)
                self._emit(f"ST    {reg} @result")

        elif isinstance(node, ast.Pass):
            pass

        else:
            raise TranspileError(
                f"Unsupported statement: {type(node).__name__}", node)

    # ─── Value extraction helpers ─────────────────────────────────────────────

    def _extract_list_values(self, node) -> list | None:
        """Extract numeric values from a list literal, supporting 1D and 2D."""
        if isinstance(node, ast.List):
            result = []
            for elt in node.elts:
                if isinstance(elt, ast.List):
                    # 2D array
                    row = self._extract_list_values(elt)
                    if row is None:
                        return None
                    result.append(row)
                elif isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    result.append(float(elt.value))
                elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                    if isinstance(elt.operand, ast.Constant):
                        result.append(-float(elt.operand.value))
                    else:
                        return None
                else:
                    return None
            return result
        return None

    def _extract_shape(self, node) -> tuple:
        """Extract shape from a tuple/list/int constant."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return (node.value,)
        if isinstance(node, (ast.Tuple, ast.List)):
            dims = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    dims.append(1)
            return tuple(dims)
        return (1,)

    def _extract_shape_from_args(self, args) -> tuple:
        """Extract shape from reshape args: x.reshape(3, 2) or x.reshape((3,2))."""
        if len(args) == 1:
            return self._extract_shape(args[0])
        return tuple(
            a.value if isinstance(a, ast.Constant) else 1
            for a in args
        )

    # ─── Main entry point ─────────────────────────────────────────────────────

    def convert(self):
        """Parse and convert the Python source to NML."""
        try:
            tree = ast.parse(self.source, filename=self.filename)
        except SyntaxError as e:
            raise TranspileError(f"Python syntax error: {e}")

        # Header
        self.instructions.append(f"; NML program generated by nml_from_numpy.py")
        self.instructions.append(f"; Source: {self.filename}")
        self.instructions.append(f"")

        # First pass: collect function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.func_map[node.name] = node

        # Second pass: compile top-level statements
        for stmt in tree.body:
            try:
                self._compile_statement(stmt)
            except TranspileError:
                raise
            except Exception as e:
                lineno = getattr(stmt, 'lineno', '?')
                raise TranspileError(f"Internal error at line {lineno}: {e}", stmt)

        # Store any variables that look like outputs
        output_candidates = [
            name for name, reg in self.var_map.items()
            if reg.startswith('R') and name not in ('i', 'j', 'k', 'n', '_')
        ]
        if output_candidates:
            self.instructions.append("")
            self._emit_comment("Store results")
            for name in output_candidates:
                reg = self.var_map[name]
                if reg.startswith('R'):
                    self._emit(f"ST    {reg} @{self._safe_name(name)}")

        self.instructions.append("HALT")
        return self

    def generate_nml(self) -> str:
        return '\n'.join(self.instructions) + '\n'

    def generate_data(self) -> str:
        if not self.data_tensors:
            return ''
        lines = []
        for name, (shape, values) in self.data_tensors.items():
            shape_str = ','.join(str(s) for s in shape)
            data_str = ','.join(f'{v:.6g}' for v in values)
            lines.append(f'@{name} shape={shape_str} dtype=f64 data={data_str}')
        return '\n'.join(lines) + '\n'


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Convert Python/NumPy scripts to NML programs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Python/NumPy subset:
  np.array, np.zeros, np.ones, np.dot, np.tanh, np.maximum,
  np.sum, np.mean, np.reshape, np.transpose, np.clip, np.softmax
  Arithmetic: +, -, *, /, @, **2
  Control flow: for x in range(N), if/else
  Patterns: 1/(1+np.exp(-x)) → SIGM, np.maximum(0,x) → RELU

Examples:
  python3 transpilers/nml_from_numpy.py script.py
  python3 transpilers/nml_from_numpy.py script.py --output-nml out.nml --output-data out.nml.data
  python3 transpilers/nml_from_numpy.py script.py --dry-run --verbose
""")
    parser.add_argument('input', help='Input Python file')
    parser.add_argument('--output-nml', help='Output .nml file (default: input.nml)')
    parser.add_argument('--output-data', help='Output .nml.data file (default: input.nml.data)')
    parser.add_argument('--dry-run', action='store_true', help='Print output without writing files')
    parser.add_argument('--verbose', action='store_true', help='Include comments mapping Python vars to registers')
    args = parser.parse_args()

    input_path = Path(args.input)
    out_nml = Path(args.output_nml) if args.output_nml else input_path.with_suffix('.nml')
    out_data = Path(args.output_data) if args.output_data else input_path.with_suffix('.nml.data')

    source = input_path.read_text(encoding='utf-8')

    print(f"Transpiling {input_path}...")
    converter = NumpyToNML(source, filename=str(input_path), verbose=args.verbose)

    try:
        converter.convert()
    except TranspileError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    nml_text = converter.generate_nml()
    data_text = converter.generate_data()

    if args.dry_run:
        print("\n=== NML Program ===")
        print(nml_text)
        if data_text:
            print(f"=== Data ({len(converter.data_tensors)} tensors) ===")
            print(data_text)
    else:
        out_nml.write_text(nml_text, encoding='utf-8')
        print(f"Written: {out_nml}")
        if data_text:
            out_data.write_text(data_text, encoding='utf-8')
            print(f"Written: {out_data} ({len(converter.data_tensors)} tensors)")
        else:
            print(f"No data file needed (no array literals)")

    if converter.warnings:
        print(f"\nWarnings ({len(converter.warnings)}):")
        for w in converter.warnings:
            print(f"  [WARN]  {w}")

    instr_count = sum(1 for line in converter.instructions
                      if line.strip() and not line.strip().startswith(';'))
    print(f"\nDone. {instr_count} instructions generated.")


if __name__ == '__main__':
    main()
