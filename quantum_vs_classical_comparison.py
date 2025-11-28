"""
=====================================================================
COMPARACIÓN: ALGORITMO DE GROVER (CUÁNTICO) VS BÚSQUEDA CLÁSICA
=====================================================================

Este archivo implementa y compara el algoritmo de búsqueda de Grover
(simulado cuánticamente) con la búsqueda lineal clásica.

Contenido:
1. Búsqueda Clásica Lineal
2. Simulación del Algoritmo de Grover
3. Implementación con Qiskit (opcional)
4. Benchmarking y Comparación
5. Visualización de Resultados

=====================================================================
"""

import time
import math
import random
from typing import Tuple, List, Dict
from statistics import mean
import numpy as np

# Imports opcionales para Qiskit (comentar si no se usa)
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Diagonal
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =====================================================================
# SECCIÓN 1: BÚSQUEDA CLÁSICA
# =====================================================================

def linear_search(arr: List[int], target: int) -> Tuple[int, int, float]:
    """
    Búsqueda lineal clásica en un arreglo.
    
    Args:
        arr: Lista de enteros donde buscar
        target: Valor a buscar
        
    Returns:
        Tupla con (índice encontrado o -1, número de comparaciones, tiempo en segundos)
    """
    start = time.perf_counter()
    comparisons = 0
    
    for i, v in enumerate(arr):
        comparisons += 1
        if v == target:
            elapsed = time.perf_counter() - start
            return i, comparisons, elapsed
    
    elapsed = time.perf_counter() - start
    return -1, comparisons, elapsed


def generate_array(n: int) -> List[int]:
    """Genera un arreglo con valores de 0 a n-1."""
    return list(range(n))


# =====================================================================
# SECCIÓN 2: ALGORITMO DE GROVER (SIMULACIÓN)
# =====================================================================

def grover_simulation(N: int, target: int) -> Tuple[int, float, float]:
    """
    Implementa el algoritmo de Grover usando Qiskit.
    
    El algoritmo de Grover proporciona una aceleración cuadrática:
    - Clásico: O(N) comparaciones
    - Grover: O(√N) iteraciones
    
    Args:
        N: Tamaño del espacio de búsqueda (debe ser potencia de 2)
        target: Índice del elemento a buscar
        
    Returns:
        Tupla con (iteraciones, probabilidad de éxito, tiempo en segundos)
    """
    if N & (N - 1) != 0:
        raise ValueError("N debe ser una potencia de 2")
    
    start = time.perf_counter()
    
    # Calcular número de qubits necesarios
    n_qubits = int(math.log2(N))
    
    # Número óptimo de iteraciones de Grover: π/4 * √N
    iterations = max(1, int(math.floor((math.pi / 4) * math.sqrt(N))))
    
    # Crear el circuito de Grover usando Qiskit
    if QISKIT_AVAILABLE:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import DiagonalGate
        from qiskit.quantum_info import Statevector
        
        # Construir el circuito
        qc = QuantumCircuit(n_qubits)
        
        # Inicialización: Hadamard en todos los qubits (superposición)
        for q in range(n_qubits):
            qc.h(q)
        
        # Crear oráculo diagonal
        phases = [1.0] * N
        phases[target] = -1.0
        oracle_gate = DiagonalGate(phases)
        
        # Crear operador de difusión
        d0 = [-1.0] * N
        d0[0] = 1.0
        diffusion_gate = DiagonalGate(d0)
        
        # Aplicar iteraciones de Grover
        for _ in range(iterations):
            # Aplicar oráculo
            qc.append(oracle_gate, range(n_qubits))
            
            # Aplicar difusión: H -> D0 -> H
            for q in range(n_qubits):
                qc.h(q)
            qc.append(diffusion_gate, range(n_qubits))
            for q in range(n_qubits):
                qc.h(q)
        
        # Simular con Statevector
        init_state = Statevector.from_label('0' * n_qubits)
        final_state = init_state.evolve(qc)
        probs = final_state.probabilities_dict()
        
        # Obtener probabilidad del estado objetivo
        target_binary = format(target, 'b').zfill(n_qubits)
        probability = probs.get(target_binary, 0.0)
    else:
        # Fallback a simulación con NumPy si Qiskit no está disponible
        psi = np.ones(N, dtype=np.complex128) / math.sqrt(N)
        oracle = np.ones(N, dtype=np.complex128)
        oracle[target] = -1.0
        
        for _ in range(iterations):
            psi = oracle * psi
            mean_amplitude = np.sum(psi) / N
            psi = 2 * mean_amplitude - psi
        
        probability = abs(psi[target]) ** 2
    
    elapsed = time.perf_counter() - start
    
    return iterations, float(probability), elapsed


# =====================================================================
# SECCIÓN 3: IMPLEMENTACIÓN CON QISKIT (OPCIONAL)
# =====================================================================

if QISKIT_AVAILABLE:
    
    def diagonal_oracle(n_qubits: int, target: int) -> QuantumCircuit:
        """
        Crea un oráculo diagonal que aplica fase -1 al estado objetivo.
        
        Args:
            n_qubits: Número de qubits
            target: Estado objetivo (entero de 0 a 2^n_qubits - 1)
            
        Returns:
            QuantumCircuit con el oráculo
        """
        from qiskit.circuit.library import DiagonalGate
        
        N = 2 ** n_qubits
        phases = [1.0] * N
        phases[target] = -1.0
        
        diag = DiagonalGate(phases)
        qc = QuantumCircuit(n_qubits)
        qc.append(diag, range(n_qubits))
        return qc
    
    
    def diffusion_operator(n_qubits: int) -> QuantumCircuit:
        """
        Construye el operador de difusión: D = 2|s⟩⟨s| - I
        
        Implementación: H^⊗n · (2|0⟩⟨0| - I) · H^⊗n
        
        Args:
            n_qubits: Número de qubits
            
        Returns:
            QuantumCircuit con el operador de difusión
        """
        from qiskit.circuit.library import DiagonalGate
        
        N = 2 ** n_qubits
        
        # Crear diagonal D0 = 2|0⟩⟨0| - I
        d0 = [-1.0] * N
        d0[0] = 1.0
        
        qc = QuantumCircuit(n_qubits)
        
        # Aplicar Hadamard en todos los qubits
        for q in range(n_qubits):
            qc.h(q)
        
        # Aplicar D0
        qc.append(DiagonalGate(d0), range(n_qubits))
        
        # Aplicar Hadamard nuevamente
        for q in range(n_qubits):
            qc.h(q)
        
        return qc
    
    
    def grover_circuit(n_qubits: int, target: int, iterations: int) -> QuantumCircuit:
        """
        Construye el circuito completo de Grover.
        
        Args:
            n_qubits: Número de qubits
            target: Estado objetivo
            iterations: Número de iteraciones de Grover
            
        Returns:
            QuantumCircuit con el algoritmo de Grover completo
        """
        qc = QuantumCircuit(n_qubits)
        
        # Inicialización: superposición uniforme
        for q in range(n_qubits):
            qc.h(q)
        
        # Crear oráculo y difusión
        oracle = diagonal_oracle(n_qubits, target)
        diffusion = diffusion_operator(n_qubits)
        
        # Aplicar iteraciones de Grover
        for _ in range(iterations):
            qc.append(oracle.to_instruction(), range(n_qubits))
            qc.append(diffusion.to_instruction(), range(n_qubits))
        
        return qc
    
    
    def run_grover_qiskit(n_qubits: int, target: int) -> Tuple[int, float, Dict]:
        """
        Ejecuta el algoritmo de Grover usando Qiskit.
        
        Args:
            n_qubits: Número de qubits
            target: Estado objetivo
            
        Returns:
            Tupla con (iteraciones, probabilidad del target, diccionario de probabilidades)
        """
        N = 2 ** n_qubits
        iterations = max(1, int(math.floor((math.pi / 4) * math.sqrt(N))))
        
        qc = grover_circuit(n_qubits, target, iterations)
        
        # Simular con Statevector
        init = Statevector.from_label('0' * n_qubits)
        final = init.evolve(qc)
        probs = final.probabilities_dict()
        
        # Obtener probabilidad del estado objetivo
        target_binary = format(target, 'b').zfill(n_qubits)
        p_target = probs.get(target_binary, 0.0)
        
        return iterations, p_target, probs


# =====================================================================
# SECCIÓN 4: BENCHMARKING Y COMPARACIÓN
# =====================================================================

def run_single_comparison(N: int, target: int) -> Dict:
    """
    Ejecuta una comparación única entre búsqueda clásica y Grover.
    
    Args:
        N: Tamaño del espacio de búsqueda
        target: Elemento a buscar
        
    Returns:
        Diccionario con resultados de ambos métodos
    """
    # Búsqueda clásica
    arr = generate_array(N)
    idx, comparisons, time_classical = linear_search(arr, target)
    
    # Grover simulado
    iterations, probability, time_grover = grover_simulation(N, target)
    
    return {
        'N': N,
        'target': target,
        'classical': {
            'index': idx,
            'comparisons': comparisons,
            'time_ms': time_classical * 1000
        },
        'grover': {
            'iterations': iterations,
            'probability': probability,
            'time_ms': time_grover * 1000
        },
        'speedup_factor': comparisons / iterations if iterations > 0 else 0
    }


def run_benchmarks(Ns: List[int] = None, trials: int = 20) -> Dict:
    """
    Ejecuta múltiples benchmarks y calcula estadísticas.
    
    Args:
        Ns: Lista de tamaños a probar (potencias de 2)
        trials: Número de pruebas por tamaño
        
    Returns:
        Diccionario con datos agregados
    """
    if Ns is None:
        Ns = [8, 16, 32, 64, 128]
    
    data = {
        'N': [],
        'classical_comparisons': [],
        'grover_iterations': [],
        'grover_probability': [],
        'speedup_factor': [],
        'theoretical_speedup': []
    }
    
    print("\n" + "="*80)
    print("BENCHMARK: BÚSQUEDA CLÁSICA VS ALGORITMO DE GROVER")
    print("="*80)
    print(f"{'N':<10}{'Comps':<15}{'Iters':<15}{'Speedup':<15}{'Teórico':<15}{'Prob':<10}")
    print("-"*80)
    
    for N in Ns:
        classical_comps = []
        grover_iters = []
        grover_probs = []
        speedups = []
        
        for _ in range(trials):
            target = random.randrange(N)
            result = run_single_comparison(N, target)
            
            classical_comps.append(result['classical']['comparisons'])
            grover_iters.append(result['grover']['iterations'])
            grover_probs.append(result['grover']['probability'])
            speedups.append(result['speedup_factor'])
        
        # Calcular promedios
        avg_classical_comps = mean(classical_comps)
        avg_grover_iters = mean(grover_iters)
        avg_grover_prob = mean(grover_probs)
        avg_speedup = mean(speedups)
        theoretical = math.sqrt(N)
        
        data['N'].append(N)
        data['classical_comparisons'].append(avg_classical_comps)
        data['grover_iterations'].append(avg_grover_iters)
        data['grover_probability'].append(avg_grover_prob)
        data['speedup_factor'].append(avg_speedup)
        data['theoretical_speedup'].append(theoretical)
        
        print(f"{N:<10}{avg_classical_comps:<15.1f}{avg_grover_iters:<15.1f}"
              f"{avg_speedup:<15.2f}{theoretical:<15.2f}{avg_grover_prob:<10.4f}")
    
    print("="*80)
    print(f"Pruebas por N: {trials} | Speedup = Comparaciones / Iteraciones")
    print("="*80 + "\n")
    
    return data


# =====================================================================
# SECCIÓN 5: VISUALIZACIÓN DE RESULTADOS
# =====================================================================

if MATPLOTLIB_AVAILABLE:
    
    def plot_comparisons(data: Dict, output_path: str = 'comparisons_vs_N.png'):
        """Grafica comparaciones clásicas vs iteraciones de Grover."""
        Ns = data['N']
        
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, data['classical_comparisons'], marker='o', linewidth=2,
                 label='Comparaciones Clásicas (O(N))', color='red')
        plt.plot(Ns, data['grover_iterations'], marker='s', linewidth=2,
                 label='Iteraciones Grover (O(√N))', color='blue')
        
        plt.xscale('log', base=2)
        plt.xlabel('N (tamaño del espacio de búsqueda, escala log₂)', fontsize=12)
        plt.ylabel('Número de operaciones', fontsize=12)
        plt.title('Comparación: Búsqueda Clásica vs Algoritmo de Grover', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✓ Gráfica guardada: {output_path}")
    
    
    def plot_speedup(data: Dict, output_path: str = 'speedup_vs_N.png'):
        """Grafica el factor de aceleración (speedup) de Grover."""
        Ns = data['N']
        
        # Calcular speedup teórico (√N)
        theoretical_speedup = [math.sqrt(N) for N in Ns]
        
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, data['speedup_factor'], marker='o', linewidth=2,
                 label='Speedup Observado', color='green')
        plt.plot(Ns, theoretical_speedup, linestyle='--', linewidth=2,
                 label='Speedup Teórico (√N)', color='orange')
        
        plt.xscale('log', base=2)
        plt.xlabel('N (tamaño del espacio de búsqueda, escala log₂)', fontsize=12)
        plt.ylabel('Factor de Aceleración (Speedup)', fontsize=12)
        plt.title('Aceleración Cuadrática del Algoritmo de Grover', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✓ Gráfica guardada: {output_path}")
    
    
    def plot_complexity(data: Dict, output_path: str = 'complexity_vs_N.png'):
        """Grafica la complejidad computacional teórica."""
        Ns = data['N']
        
        # Calcular complejidades teóricas
        linear_theory = Ns  # O(N)
        sqrt_theory = [math.sqrt(N) for N in Ns]  # O(√N)
        
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, linear_theory, linestyle='--', linewidth=2, alpha=0.7,
                 label='O(N) - Clásico (teórico)', color='red')
        plt.plot(Ns, data['classical_comparisons'], marker='o', linewidth=2,
                 label='Clásico (observado)', color='darkred')
        plt.plot(Ns, sqrt_theory, linestyle='--', linewidth=2, alpha=0.7,
                 label='O(√N) - Grover (teórico)', color='blue')
        plt.plot(Ns, data['grover_iterations'], marker='s', linewidth=2,
                 label='Grover (observado)', color='darkblue')
        
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlabel('N (tamaño del espacio de búsqueda, escala log₂)', fontsize=12)
        plt.ylabel('Número de Operaciones (escala log₂)', fontsize=12)
        plt.title('Complejidad Computacional: O(N) vs O(√N)', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✓ Gráfica guardada: {output_path}")
    
    
    def plot_probability(data: Dict, output_path: str = 'probability_vs_N.png'):
        """Grafica la probabilidad de éxito de Grover."""
        Ns = data['N']
        
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, data['grover_probability'], marker='o', linewidth=2,
                 color='purple', label='Probabilidad de Éxito')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Probabilidad Ideal (1.0)')
        
        plt.xscale('log', base=2)
        plt.ylim(0, 1.05)
        plt.xlabel('N (tamaño del espacio de búsqueda, escala log₂)', fontsize=12)
        plt.ylabel('Probabilidad de Éxito', fontsize=12)
        plt.title('Probabilidad de Éxito del Algoritmo de Grover', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✓ Gráfica guardada: {output_path}")
    
    
    def generate_all_plots(data: Dict):
        """Genera todas las gráficas de comparación."""
        print("\n" + "="*70)
        print("GENERANDO VISUALIZACIONES")
        print("="*70)
        
        plot_comparisons(data, 'comparisons_vs_N.png')
        plot_speedup(data, 'speedup_vs_N.png')
        plot_complexity(data, 'complexity_vs_N.png')
        plot_probability(data, 'probability_vs_N.png')
        
        print("="*70 + "\n")


# =====================================================================
# SECCIÓN 6: FUNCIÓN PRINCIPAL
# =====================================================================

def main():
    """Función principal que ejecuta el análisis completo."""
    
    print("\n" + "="*80)
    print("     COMPARACIÓN: ALGORITMO DE GROVER VS BÚSQUEDA CLÁSICA")
    print("="*80)
    print("\n📚 TEORÍA - VENTAJA CUÁNTICA:")
    print("   • Búsqueda Clásica: Complejidad O(N) - revisar cada elemento")
    print("   • Algoritmo de Grover: Complejidad O(√N) - amplificación cuántica")
    print("   • Aceleración Cuadrática: Factor de speedup ~√N")
    print("   • Ejemplo: Para N=1,000,000 elementos")
    print("      - Clásico: ~500,000 comparaciones promedio")
    print("      - Grover: ~785 iteraciones (¡636x más rápido!)")
    print("\n" + "="*80)
    
    # Configuración
    Ns = [8, 16, 32, 64, 128, 256]  # Tamaños a probar
    trials = 30  # Número de pruebas por tamaño
    
    # Ejecutar benchmarks
    data = run_benchmarks(Ns, trials)
    
    # Generar visualizaciones
    if MATPLOTLIB_AVAILABLE:
        generate_all_plots(data)
    else:
        print("\n⚠ Matplotlib no disponible. No se generarán gráficas.")
        print("  Instala con: pip install matplotlib")
    
    # Ejemplo con Qiskit (si está disponible)
    if QISKIT_AVAILABLE:
        print("\n" + "="*70)
        print("EJEMPLO CON QISKIT")
        print("="*70)
        for n in [3, 4, 5]:
            N = 2 ** n
            target = N // 3
            iters, p_target, probs = run_grover_qiskit(n, target)
            print(f"n_qubits={n}, N={N}, target={target}, "
                  f"iterations={iters}, probability={p_target:.6f}")
        print("="*70 + "\n")
    else:
        print("\n⚠ Qiskit no disponible. Usa: pip install qiskit")
    
    print("\n✅ Análisis completo finalizado.\n")


# =====================================================================
# EJEMPLOS DE USO
# =====================================================================

def example_basic_usage():
    """Ejemplo básico de uso."""
    print("\n=== EJEMPLO BÁSICO ===\n")
    
    N = 64
    target = 42
    
    # Búsqueda clásica
    arr = generate_array(N)
    idx, comps, time_c = linear_search(arr, target)
    print(f"Búsqueda Clásica:")
    print(f"  N = {N}, target = {target}")
    print(f"  Comparaciones: {comps}")
    print(f"  Tiempo: {time_c*1000:.4f} ms\n")
    
    # Grover
    iters, prob, time_g = grover_simulation(N, target)
    print(f"Algoritmo de Grover:")
    print(f"  N = {N}, target = {target}")
    print(f"  Iteraciones: {iters}")
    print(f"  Probabilidad de éxito: {prob:.6f}")
    print(f"  Tiempo: {time_g*1000:.4f} ms")
    print(f"  Speedup: {comps/iters:.2f}x\n")


# =====================================================================
# PUNTO DE ENTRADA
# =====================================================================

if __name__ == "__main__":
    # Descomentar para ejecutar el análisis completo
    main()
    
    # Descomentar para ver solo el ejemplo básico
    # example_basic_usage()
