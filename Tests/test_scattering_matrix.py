import unittest
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

import jax
jax.config.update("jax_enable_x64", True)

from autolattice.scattering import calc_scattering_matrix, calc_scattering_matrix_1D_chain, prepare_parameters_1D_chain, Multimode_system

class Scattering_Test(unittest.TestCase):
    def test_single_mode(self):
        # single mode, see analytical calculations 9.1.23

        N_modes = 1
        Omega = 1.
        kappa = 0.1
        g = 0.1 + 0.37j  # absolute value has to be below 1./2.*np.sqrt(kappa**2+Omega**2), such that system is stable
        num_omegas = 500
        omegas = jnp.linspace(-2., 3., num_omegas)
        mode_freqs = jnp.array([Omega])
        on_site_adag_adag = jnp.array([g])
        coupling_adag_a = jnp.array([])  # J
        coupling_adag_adag = jnp.array([])  # L
        int_losses = jnp.array([0]) * 2
        ext_losses = jnp.array([kappa]) * 2

        scattering_matrix, info_dict = calc_scattering_matrix_1D_chain(omegas, mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses)

        expected = np.full([len(omegas), 1, 1], -100, dtype='complex')
        expected[:, 0, 0] = 1. - 1.j * 2 * kappa / (
                omegas - Omega + 1.j * kappa + 4 * np.abs(g) ** 2 / (omegas + Omega + 1.j * kappa))

        fig, ax = plt.subplots(N_modes, N_modes, figsize=(4, 3))
        ax.set_title('$S$')
        ax.set_xlabel('$\omega$')
        ax.plot(omegas, np.real(scattering_matrix[:, 0, 0]), label='Re')
        ax.plot(omegas, np.real(expected[:, 0, 0]), ls='dashed', color='black', label='theory')
        ax.plot(omegas, np.imag(scattering_matrix[:, 0, 0]), label='Im')
        ax.plot(omegas, np.imag(expected[:, 0, 0]), ls='dashed', color='black')
        ax.legend()
        fig.tight_layout()
        plt.savefig('Tests/scattering_single_mode.pdf')

        np.testing.assert_array_almost_equal(scattering_matrix[:, :1, :1], expected, decimal=5)
        self.assertEqual(True, info_dict['stability'])


    def test_single_unstable_mode(self):
        # single mode, see analytical calculations 9.1.23

        N_modes = 1
        Omega = 1.
        kappa = 0.1
        g = 0.7  # absolute value has to be below 1./2.*np.sqrt(kappa**2+Omega**2), such that system is stable
        num_omegas = 500
        omegas = jnp.linspace(-2., 3., num_omegas)
        mode_freqs = jnp.array([Omega])
        on_site_adag_adag = jnp.array([g])
        coupling_adag_a = jnp.array([])  # J
        coupling_adag_adag = jnp.array([])  # L
        int_losses = jnp.array([0]) * 2
        ext_losses = jnp.array([kappa]) * 2

        scattering_matrix, info_dict = calc_scattering_matrix_1D_chain(omegas, mode_freqs, on_site_adag_adag, coupling_adag_a,
                                                              coupling_adag_adag, int_losses, ext_losses)

        expected = np.full([len(omegas), 1, 1], -100, dtype='complex')
        expected[:, 0, 0] = 1. - 1.j * 2 * kappa / (
                omegas - Omega + 1.j * kappa + 4 * np.abs(g) ** 2 / (omegas + Omega + 1.j * kappa))

        np.testing.assert_array_almost_equal(scattering_matrix[:, :1, :1], expected, decimal=5)
        self.assertEqual(False, info_dict['stability'])


    def test_phonon_photon_translator(self):
        # Compare own code with analytical results from Safavi-Naeini & Painter New J. Phys. 13 013017
        # be aware, that that the phase in Eq. (45) and (46) is not correct, see own calculations 11.1.23

        N_modes = 2
        Delta = 1.
        G = 0.3 + 0.2j
        kappa_i = 0.1
        kappa_e = 0.2
        gamma_i = 0.05
        gamma_e = 0.08
        num_omegas = 500
        omegas = jnp.linspace(-3., 2., num_omegas)
        mode_freqs = jnp.array([0, -Delta])
        on_site_adag_adag = jnp.zeros(N_modes, dtype='complex')
        coupling_adag_a = jnp.array([G, G])  # J
        coupling_adag_adag = jnp.zeros(N_modes)  # L

        int_losses = jnp.array([kappa_i, gamma_i]) * 2
        ext_losses = jnp.array([kappa_e, gamma_e]) * 2

        kappa = kappa_i + kappa_e
        gamma = gamma_i + gamma_e

        denom = np.abs(G) ** 2 + (gamma - 1.j * (omegas + Delta)) * (kappa - 1.j * omegas)
        expected = np.full([len(omegas), 2, 2], -100, dtype='complex')
        expected[:, 0, 0] = 1 - 2 * kappa_e * (gamma - 1.j * (Delta + omegas)) / denom
        # 0,1 and 1,0 element: phase in paper of both elements is not correct, see own calculations
        expected[:, 0, 1] = 2.j * G * np.sqrt(gamma_e * kappa_e) / denom
        expected[:, 1, 0] = 2.j * np.conj(G) * np.sqrt(gamma_e * kappa_e) / denom
        expected[:, 1, 1] = 1 - 2 * gamma_e * (kappa - 1.j * omegas) / denom

        scattering_matrix, info_dict = calc_scattering_matrix_1D_chain(
            omegas, mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses
        )

        fig, axes = plt.subplots(N_modes, N_modes)
        for i in range(N_modes):
            for j in range(N_modes):
                ax = axes[i, j]
                ax.set_title('$S_{%i%i}$' % (i, j))
                ax.set_xlabel('$\omega$')
                ax.plot(omegas, np.real(scattering_matrix[:, i, j]), label='Re')
                ax.plot(omegas, np.real(expected[:, i, j]), ls='dashed', color='black', label='theory')
                ax.plot(omegas, np.imag(scattering_matrix[:, i, j]), label='Im')
                ax.plot(omegas, np.imag(expected[:, i, j]), ls='dashed', color='black')
        axes[0, 0].legend()
        fig.tight_layout()
        plt.savefig('Tests/scattering_phonon_photon_translator.pdf')

        np.testing.assert_array_almost_equal(scattering_matrix[:, :N_modes, :N_modes], expected)
        self.assertEqual(True, info_dict['stability'])

    def test_quantum_state_transfer(self):
        # Compare own code with analytical results from Wang & Clerk PRL 108, 153603 (2012)
        N_modes = 3
        omegam = 0.5
        Delta1 = 1.
        Delta2 = 2.
        G1 = 0.1
        G2 = 0.3
        # intrinsic error rates are 0
        kappa1 = 0.05
        kappa2 = 0.03
        gamma = 0.08
        int_losses = jnp.zeros(N_modes)
        ext_losses = jnp.array([kappa1, gamma, kappa2])

        num_omegas = 500
        omegas = jnp.linspace(-2.5, -1.5, num_omegas)

        mode_freqs = jnp.array([-Delta1, omegam, -Delta2])
        on_site_adag_adag = jnp.zeros(N_modes)
        coupling_adag_a = jnp.array([G1, G2, 0])  # J
        coupling_adag_adag = jnp.zeros(N_modes)  # L

        scattering_matrix, info_dict = calc_scattering_matrix_1D_chain(
            omegas, mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses
        )

        mu1 = kappa1 / 2. / (kappa1 / 2. - 1.j * (omegas + Delta1))
        mu2 = kappa2 / 2. / (kappa2 / 2. - 1.j * (omegas + Delta2))
        mum = gamma / 2. / (gamma / 2. - 1.j * (omegas - omegam))
        C1 = G1 ** 2 / (gamma * kappa1)
        C2 = G2 ** 2 / (gamma * kappa2)
        zeta = (1 / (8 * mum) + 1. / 2. * (C1 * mu1 + C2 * mu2)) ** (-1)

        expected = np.zeros([num_omegas, N_modes, N_modes], dtype='complex')
        expected[:, 0, 0] = 1 - 2 * mu1 + C1 * zeta * mu1 ** 2
        expected[:, 1, 1] = 1 - 1. / 4. * zeta
        expected[:, 2, 2] = 1 - 2 * mu2 + C2 * zeta * mu2 ** 2
        expected[:, 0, 2] = expected[:, 2, 0] = np.sqrt(C1 * C2) * mu1 * mu2 * zeta
        expected[:, 1, 0] = expected[:, 0, 1] = 1.j / 2. * np.sqrt(C1) * mu1 * zeta
        expected[:, 1, 2] = expected[:, 2, 1] = 1.j / 2. * np.sqrt(C2) * mu2 * zeta

        fig, axes = plt.subplots(N_modes, N_modes, figsize=(8, 6))
        for i in range(N_modes):
            for j in range(N_modes):
                ax = axes[i, j]
                ax.set_title('$S_{%i%i}$' % (i, j))
                ax.set_xlabel('$\omega$')
                ax.plot(omegas, np.real(scattering_matrix[:, i, j]), label='Re')
                ax.plot(omegas, np.real(expected[:, i, j]), ls='dashed', color='black', label='theory')
                ax.plot(omegas, np.imag(scattering_matrix[:, i, j]), label='Im')
                ax.plot(omegas, np.imag(expected[:, i, j]), ls='dashed', color='black')
                # ax.plot(omegas, np.imag(scattering[:,i,j]), ls='dashed', color='C0')
        axes[0, 0].legend()
        fig.tight_layout()
        plt.savefig('Tests/scattering_quantum_state_transfer.pdf')

        np.testing.assert_array_almost_equal(scattering_matrix[:, :N_modes, :N_modes], expected)
        self.assertEqual(True, info_dict['stability'])

    def test_quantum_state_transfer_different_order_of_modes(self):
        # function identical with test_quantum_state_transfer, only order of modes was changed
        # order here: a2, a1, d

        N_modes = 3
        omegam = 0.5
        Delta1 = 1.
        Delta2 = 2.
        G1 = 0.1
        G2 = 0.3
        # intrinsic error rates are 0
        kappa1 = 0.05
        kappa2 = 0.03
        gamma = 0.08
        int_losses = jnp.zeros(N_modes)
        ext_losses = jnp.array([kappa2, kappa1, gamma])

        num_omegas = 500
        omegas = jnp.linspace(-2.5, -1.5, num_omegas)

        mode_freqs = jnp.array([-Delta2, -Delta1, omegam])
        on_site_adag_adag = jnp.zeros(N_modes)
        coupling_adag_a = jnp.array([0, G1, G2])  # J
        coupling_adag_adag = jnp.zeros(N_modes)  # L

        scattering_matrix, info_dict = calc_scattering_matrix_1D_chain(
            omegas, mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses
        )

        mu1 = kappa1 / 2. / (kappa1 / 2. - 1.j * (omegas + Delta1))
        mu2 = kappa2 / 2. / (kappa2 / 2. - 1.j * (omegas + Delta2))
        mum = gamma / 2. / (gamma / 2. - 1.j * (omegas - omegam))
        C1 = G1 ** 2 / (gamma * kappa1)
        C2 = G2 ** 2 / (gamma * kappa2)
        zeta = (1 / (8 * mum) + 1. / 2. * (C1 * mu1 + C2 * mu2)) ** (-1)

        expected = np.zeros([num_omegas, N_modes, N_modes], dtype='complex')
        expected[:, 1, 1] = 1 - 2 * mu1 + C1 * zeta * mu1 ** 2
        expected[:, 2, 2] = 1 - 1. / 4. * zeta
        expected[:, 0, 0] = 1 - 2 * mu2 + C2 * zeta * mu2 ** 2
        expected[:, 1, 0] = expected[:, 0, 1] = np.sqrt(C1 * C2) * mu1 * mu2 * zeta
        expected[:, 2, 1] = expected[:, 1, 2] = 1.j / 2. * np.sqrt(C1) * mu1 * zeta
        expected[:, 2, 0] = expected[:, 0, 2] = 1.j / 2. * np.sqrt(C2) * mu2 * zeta

        fig, axes = plt.subplots(N_modes, N_modes, figsize=(8, 6))
        for i in range(N_modes):
            for j in range(N_modes):
                ax = axes[i, j]
                ax.set_title('$S_{%i%i}$' % (i, j))
                ax.set_xlabel('$\omega$')
                ax.plot(omegas, np.real(scattering_matrix[:, i, j]), label='Re')
                ax.plot(omegas, np.real(expected[:, i, j]), ls='dashed', color='black', label='theory')
                ax.plot(omegas, np.imag(scattering_matrix[:, i, j]), label='Im')
                ax.plot(omegas, np.imag(expected[:, i, j]), ls='dashed', color='black')
                # ax.plot(omegas, np.imag(scattering[:,i,j]), ls='dashed', color='C0')
        axes[0, 0].legend()
        fig.tight_layout()
        plt.savefig('Tests/scattering_quantum_state_transfer_different_order.pdf')

        np.testing.assert_array_almost_equal(scattering_matrix[:, :N_modes, :N_modes], expected)
        self.assertEqual(True, info_dict['stability'])

    def test_unidirectional_amplifier(self):
        # Compare own code with arxiv 2207.13728

        N_modes = 20

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs
        gs = gc
        kappa_i = 0.
        kappa_e = 2.6 * J_abs
        Delta = 0.

        mode_freqs = jnp.full(N_modes, -Delta)
        on_site_adag_adag = jnp.full(N_modes, -gs / 2.)

        coupling_adag_a = jnp.hstack((jnp.full(N_modes - 1, J_abs * jnp.exp(1.j * J_phase)), [0.]))
        coupling_adag_adag = jnp.hstack((jnp.full(N_modes - 1, -gc), [0.]))

        int_losses = jnp.full(N_modes, kappa_i)
        ext_losses = jnp.full(N_modes, kappa_e)

        num_omegas = 201
        omegas = jnp.linspace(-2., 2., num_omegas)

        expected_gain = np.array([
            0.3206824 , 0.48366774, 0.69410584, 0.91133531, 1.12949254,
            1.34777912, 1.56608298, 1.78436625, 2.00273974, 2.22088202,
            2.43943599, 2.65767402, 2.87478453, 3.09831078, 3.30233373,
            3.54784707, 3.74151974, 3.91710694, 4.45611518
        ])

        expected_reverse_gain = np.array([
            -0.24688512, -0.7812355, -1.2053825, -1.55998516, -1.87794001,
            -2.17851542, -2.47129555, -2.76068164, -3.04861691, -3.33594094,
            -3.62301054, -3.90997619, -4.19690025, -4.48380392, -4.77072072,
            -5.05759835, -5.34382699, -5.64147713, -5.81982464
        ])

        scattering_matrix, info_dict = calc_scattering_matrix_1D_chain(
            omegas, mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses
        )

        mode_selection = [0, N_modes - 1]
        fig, axes = plt.subplots(len(mode_selection), len(mode_selection), figsize=(4, 4))
        for i, mode1 in enumerate(mode_selection):
            for j, mode2 in enumerate(mode_selection):
                ax = axes[i, j]
                ax.set_title('$S_{%i,%i}$' % (mode1, mode2))
                ax.set_xlabel('$\omega$')
                ax.plot(omegas, np.abs(scattering_matrix[:, mode1, mode2]), label='simulation')
        fig.tight_layout()
        fig.savefig('Tests/scattering_unidirectional_amplifier.pdf')

        self.assertEqual(True, info_dict['stability'])

        idx = np.argwhere(np.abs(omegas == -0.5 * J_abs))[0][0]
        gain = np.log10(np.abs(scattering_matrix[idx, 0, 1:N_modes]) ** 2)
        reverse_gain = np.log10(np.abs(scattering_matrix[idx, 1:N_modes, 0]) ** 2)

        np.testing.assert_array_almost_equal(gain, expected_gain)
        np.testing.assert_array_almost_equal(reverse_gain, expected_reverse_gain)

    def test_unidirectional_simple_circulator_identical_phases(self):

        def unpack_params(params):
            adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses = prepare_parameters_1D_chain(
                mode_freqs=jnp.zeros(N_modes),
                on_site_adag_adag=jnp.zeros(N_modes),
                coupling_adag_a=params['coupling_adag_a_abs'] * jnp.exp(1.j * params['coupling_adag_a_phase']),
                coupling_adag_adag=jnp.zeros(N_modes),
                int_losses=jnp.zeros(N_modes),
                ext_losses=jnp.ones(N_modes)
            )
            return adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses

        N_modes = 3
        test_params = {
            'coupling_adag_a_abs': jnp.ones(N_modes) * 0.5,
            'coupling_adag_a_phase': jnp.ones(N_modes) * np.pi / 2.
        }

        selection_mask = np.zeros([2 * N_modes, 2 * N_modes], dtype='bool')
        selection_mask[:N_modes, :N_modes] = True

        target = jnp.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ], dtype='complex128')

        omegas = np.linspace(-10., 10., 101)

        scattering_matrix, info_dict = calc_scattering_matrix(omegas, *unpack_params(test_params))
        scattering_matrix_selection = scattering_matrix[:, selection_mask].reshape(
            [-1, N_modes, N_modes]
        )
        deviation = jnp.real(jnp.sum((jnp.abs(scattering_matrix_selection) - target) ** 2, axis=(1, 2)))

        np.testing.assert_array_almost_equal(np.min(deviation), 0.)

    def test_unidirectional_simple_circulator_different_phases(self):

        def unpack_params(params):
            adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses = prepare_parameters_1D_chain(
                mode_freqs=jnp.zeros(N_modes),
                on_site_adag_adag=jnp.zeros(N_modes),
                coupling_adag_a=params['coupling_adag_a_abs'] * jnp.exp(1.j * params['coupling_adag_a_phase']),
                coupling_adag_adag=jnp.zeros(N_modes),
                int_losses=jnp.zeros(N_modes),
                ext_losses=jnp.ones(N_modes)
            )
            return adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses

        N_modes = 3
        test_params = {
            'coupling_adag_a_abs': jnp.ones(N_modes) * 0.5,
            'coupling_adag_a_phase': jnp.ones(N_modes) * np.pi / 2. + jnp.array([0.5, -0.7, 0.2])
        }

        selection_mask = np.zeros([2 * N_modes, 2 * N_modes], dtype='bool')
        selection_mask[:N_modes, :N_modes] = True

        target = jnp.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ], dtype='complex128')

        omegas = np.linspace(-10., 10., 101)

        scattering_matrix, info_dict = calc_scattering_matrix(omegas, *unpack_params(test_params))
        scattering_matrix_selection = scattering_matrix[:, selection_mask].reshape(
            [-1, N_modes, N_modes]
        )
        deviation = jnp.real(jnp.sum((jnp.abs(scattering_matrix_selection) - target) ** 2, axis=(1, 2)))

        np.testing.assert_array_almost_equal(np.min(deviation), 0.)


class Multimode_system_Test(unittest.TestCase):
    def test_four_modes_chain_no_gain(self):

        N_modes = 4

        mode_freqs = jnp.linspace(-4., -1., N_modes)
        coupling_adag_a_abs = jnp.arange(1, N_modes + 1)
        coupling_adag_a_phase = jnp.linspace(0.5, 1., N_modes)
        coupling_adag_a = coupling_adag_a_abs * jnp.exp(1.j * coupling_adag_a_phase)
        int_losses = jnp.linspace(6., 10., N_modes)

        system = Multimode_system(N_modes, target_modes=jnp.array([0, -1]))

        for j in range(N_modes):
            system.add_adag_a_coupling(mode_freqs[j], j, j)
            system.add_intrinsic_loss(int_losses[j], j)
        for j in range(N_modes - 1):
            system.add_adag_a_coupling(coupling_adag_a[j], j, j + 1)
        system.add_adag_a_coupling(coupling_adag_a[-1], -1, 0)

        adag_a_coupling_matrix_out, adag_adag_coupling_matrix_out, int_losses_out, ext_losses_out = system.return_system_parameters()

        adag_a_coupling_matrix_true, _, _, _ = prepare_parameters_1D_chain(
            mode_freqs=mode_freqs,
            on_site_adag_adag=jnp.zeros(N_modes),
            coupling_adag_a=coupling_adag_a,
            coupling_adag_adag=jnp.zeros(N_modes),
            int_losses=jnp.zeros(N_modes),
            ext_losses=jnp.zeros(N_modes)
        )
        np.testing.assert_array_almost_equal(adag_a_coupling_matrix_out, adag_a_coupling_matrix_true)
        np.testing.assert_array_almost_equal(int_losses_out, int_losses)
        np.testing.assert_array_almost_equal(ext_losses_out, np.array([1., 0., 0., 1.]))

    def test_single_mode_with_gain(self):
        # single mode, see analytical calculations 9.1.23 & 24.7.23

        Omega = 1.
        kappa = 2.
        g = 0.1 + 0.37j  # absolute value has to be below 1./2.*np.sqrt(kappa**2+Omega**2), such that system is stable
        num_omegas = 500
        omegas = jnp.linspace(-2., 3., num_omegas)
        
        system = Multimode_system(1)
        system.add_adag_a_coupling(Omega, 0, 0)
        system.add_adag_adag_coupling(2*g, 0, 0)
        system.ext_losses = jnp.array([2*kappa])
        
        scattering_matrix, _ = calc_scattering_matrix(omegas, *system.return_system_parameters())

        expected = 1. - 1.j * 2 * kappa / (omegas - Omega + 1.j * kappa + 4 * np.abs(g) ** 2 / (omegas + Omega + 1.j * kappa))

        np.testing.assert_array_almost_equal(scattering_matrix[:, 0, 0], expected)

    def test_unidirectional_amplifier(self):
        # Compare own code with arxiv 2207.13728

        N_modes = 20

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs
        gs = gc
        kappa_e = 2.6 * J_abs
        Delta = 0.

        num_omegas = 201
        omegas = jnp.linspace(-2., 2., num_omegas)

        system = Multimode_system(N_modes)
        for mode in range(N_modes):
            system.add_adag_a_coupling(-Delta, mode, mode)
            system.add_adag_adag_coupling(-gs, mode, mode)
        
        for mode in range(N_modes-1):
            system.add_adag_a_coupling(J_abs*jnp.exp(-1.j*J_phase), mode, mode+1)
            system.add_adag_adag_coupling(-gc, mode, mode+1)

        system.ext_losses = kappa_e * jnp.ones(N_modes)

        scattering_matrix, _ = calc_scattering_matrix(omegas, *system.return_system_parameters())

        expected_gain = np.array([
            0.3206824 , 0.48366774, 0.69410584, 0.91133531, 1.12949254,
            1.34777912, 1.56608298, 1.78436625, 2.00273974, 2.22088202,
            2.43943599, 2.65767402, 2.87478453, 3.09831078, 3.30233373,
            3.54784707, 3.74151974, 3.91710694, 4.45611518
        ])

        expected_reverse_gain = np.array([
            -0.24688512, -0.7812355, -1.2053825, -1.55998516, -1.87794001,
            -2.17851542, -2.47129555, -2.76068164, -3.04861691, -3.33594094,
            -3.62301054, -3.90997619, -4.19690025, -4.48380392, -4.77072072,
            -5.05759835, -5.34382699, -5.64147713, -5.81982464
        ])

        idx = np.argwhere(np.abs(omegas == -0.5 * J_abs))[0][0]
        gain = np.log10(np.abs(scattering_matrix[idx, 1:N_modes, 0]) ** 2)
        reverse_gain = np.log10(np.abs(scattering_matrix[idx, 0, 1:N_modes]) ** 2)

        np.testing.assert_array_almost_equal(gain, expected_gain)
        np.testing.assert_array_almost_equal(reverse_gain, expected_reverse_gain)

class Bogoliubov_Test(unittest.TestCase):
    def test_Bogoliubov_transformation(self):
        N_modes = 20

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs * jnp.exp(1.j*0.4)
        gs = 0.3 * J_abs * jnp.exp(-1.j*0.7)
        kappa_i = 0.
        kappa_e = 2.6 * J_abs
        Delta = 0.

        mode_freqs = jnp.full(N_modes, -Delta)
        on_site_adag_adag = jnp.full(N_modes, -gs / 2.)

        coupling_adag_a = jnp.hstack((jnp.full(N_modes - 1, J_abs * jnp.exp(1.j * J_phase)), [0.]))
        coupling_adag_adag = jnp.hstack((jnp.full(N_modes - 1, -gc), [0.]))

        int_losses = jnp.full(N_modes, kappa_i)
        ext_losses = jnp.full(N_modes, kappa_e)

        num_omegas = 201
        omegas = jnp.linspace(-2., 2., num_omegas)

        scattering_matrix, _ = calc_scattering_matrix_1D_chain(
            omegas, mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses
        )

        sigmaz = jnp.diag(jnp.concatenate((jnp.ones(N_modes), - jnp.ones(N_modes))))
        differences = np.sum(np.abs(sigmaz[None] - np.einsum('ijk,kl,ilm->ijm', scattering_matrix, sigmaz, jnp.conjugate(np.transpose(scattering_matrix, axes=[0,2,1])))), axis=(-2, -1))

        np.testing.assert_array_almost_equal(differences, np.zeros(num_omegas))

if __name__ == '__main__':
    unittest.main()
