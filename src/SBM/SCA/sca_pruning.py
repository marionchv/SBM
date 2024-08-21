#! /usr/bin/env python3
"""
@author: Emily Hinds
"""

from math import ceil
import numpy as np
import scipy.io as sio
import SBM.utils.utils as ut

## GLOBAL CONSTANTS ##

numToAA = dict(enumerate("-ACDEFGHIKLMNPQRSTVWY"))
AAtoNum = dict((v, k) for k, v in numToAA.items())

# add the '.' character for A2M files... annoying...
AAtoNum['.'] = AAtoNum['-']

freq0_gapless = np.array([0.073, 0.025, 0.050, 0.061, 0.042, 0.072, 0.023, 0.053,
            0.064, 0.089, 0.023, 0.043, 0.052, 0.040, 0.052, 0.073,
            0.056, 0.063, 0.013, 0.033,])

def alg2bin(alg, N_aa=20):
    """
    Translate an alignment of matrix of size M sequences by L positions where
    the amino acids are represented by numbers between 0 and N_aa (obtained
    using lett2num) to a binary array of size M x (N_aa x L).

    **Example**::

      Abin = alg2bin(alg, N_aa=20)
      
      This function is taken from pySCA.scaTools, re-included here so as not having
      to have pySCA installation as a dependency
    """

    [N_seq, N_pos] = alg.shape
    Abin_tensor = np.zeros((N_aa, N_pos, N_seq))
    for ia in range(N_aa):
        Abin_tensor[ia, :, :] = (alg == ia + 1).T
    Abin = Abin_tensor.reshape(N_aa * N_pos, N_seq, order="F").T
    return Abin

def lett2num(msa_lett, code="ACDEFGHIKLMNPQRSTVWY"):
    """
    Translate an alignment from a representation where the 20 natural amino
    acids are represented by letters to a representation where they are
    represented by the numbers 1,...,20, with any symbol not corresponding to
    an amino acid represented by 0.

    **Example**::

       msa_num = lett2num(msa_lett, code='ACDEFGHIKLMNPQRSTVWY')
       
    This function is taken from pySCA.scaTools, re-included here so as not having
    to have pySCA installation as a dependency
    """

    lett2index = {aa: i + 1 for i, aa in enumerate(code)}
    [Nseq, Npos] = [len(msa_lett), len(msa_lett[0])]
    msa_num = np.zeros((Nseq, Npos)).astype(int)
    for s, seq in enumerate(msa_lett):
        for i, lett in enumerate(seq):
            if lett in code:
                msa_num[s, i] = lett2index[lett]
    return msa_num

def seqWeights(alg, max_seqid=0.8, gaps=1, slow=False, batchsize=10000):
    """
    Compute sequence weights for an alignment (format: list of sequences) where
    the weight of a sequence is the inverse of the number of sequences in its
    neighborhood, defined as the sequences with sequence similarity below
    max_seqid. The sum of the weights defines an effective number of sequences.
    **Arguments**
      :alg: list of sequences
    **Keyword Arguments**
      :max_seqid:
      :gaps: If gaps == 1 (default), considering gaps as a 21st amino acid, if
             gaps == 0, not considering them.
    **Example**::
      seqw = seqWeights(alg)
    """

    codeaa = "ACDEFGHIKLMNPQRSTVWY"
    if gaps == 1:
        codeaa += "-"
    msa_num = lett2num(alg, code=codeaa)
    Nseq, Npos = msa_num.shape

    if not slow:
        X2d = alg2bin(msa_num, N_aa=len(codeaa))
        simMat = X2d.dot(X2d.T) / Npos
        seqw = np.array(1 / (simMat > max_seqid).sum(axis=0))
    else:
        seqw = np.zeros(Nseq)
        for i in range(0, Nseq, batchsize):
            curr_batch = msa_num[i:i+batchsize,:]
            X2d_1 = alg2bin(curr_batch, N_aa=len(codeaa))
            for j in range(0, Nseq, batchsize):
                X2d_2 = alg2bin(msa_num[j:j+batchsize], N_aa=len(codeaa))
                simMat = X2d_1.dot(X2d_2.T) / Npos
                seqw[i:i+batchsize] += np.array((simMat > max_seqid).sum(axis=1))
        seqw = 1/seqw

    seqw.shape = (1, Nseq)
    return seqw

def freq(alg, seqw=1, Naa=21, lbda=0, freq0=np.ones(21)/21): # modified from scaTools
    Nseq, Npos = alg.shape
    if isinstance(seqw, int) and seqw == 1:
        seqw = np.ones((1, Nseq))
    seqwn = (seqw / seqw.sum()).reshape((-1,))
    freq1 = np.zeros((Npos, Naa))
    freq2 = np.zeros((Npos, Npos, Naa, Naa))
    # really inefficiently calculate frequencies
    for m in range(Nseq):
        for p1 in range(Npos):
            freq1[p1, alg[m,p1]] += seqwn[m]
            for p2 in range(p1, Npos):
                freq2[p1, p2, alg[m,p1], alg[m,p2]] += seqwn[m]
                if p1 != p2:
                    freq2[p2, p1, alg[m,p2], alg[m,p1]] += seqwn[m]
                
    # background + regularization
    block = np.outer(freq0, freq0)
    freq2_bkg = np.tile(block, (Npos, Npos,1,1))
    for i in range(Npos):
        freq2_bkg[i,i] = np.diag(freq0)
        
    freq1_reg = (1-lbda)*freq1 + lbda*np.tile(freq0, (Npos,1))
    freq2_reg = (1-lbda)*freq2 + lbda*freq2_bkg
    freq0_reg = freq1_reg.mean(axis=0)
    return freq1_reg, freq2_reg, freq0_reg

def posWeights(alg,seqw=1,lbda=0,N_aa=20,freq0=np.array(
        [0.073, 0.025, 0.050, 0.061, 0.042, 0.072, 0.023, 0.053,
         0.064, 0.089, 0.023, 0.043, 0.052, 0.040, 0.052, 0.073,
         0.056, 0.063, 0.013, 0.033,]),
        tolerance=1e-12):

    N_seq, N_pos = alg.shape
    if isinstance(seqw, int) and seqw == 1:
        seqw = np.ones((1, N_seq))
    freq1, freq2, _ = freq(alg, Naa=N_aa, seqw=seqw, lbda=lbda, freq0=freq0)
    
    # Overall fraction of gaps:
    theta = 1 - freq1.sum() / N_pos
    if theta < tolerance:
        theta = 0
        
    # Background frequencies with gaps:
    freqg0 = (1 - theta) * freq0
    freq0v = np.tile(freq0, (N_pos,1))
    iok = np.array(np.where(np.logical_and(freq1 > 0, freq1 < 1))).T.tolist()
    
    # Derivatives of relative entropy per position and amino acid:
    Wia = np.zeros((N_pos, N_aa))
    Dia = np.zeros((N_pos, N_aa))
    for idx in iok:
        
        Wia[idx] = abs(
            np.log(
                (freq1[idx] * (1 - freq0v[idx])) / ((1 - freq1[idx]) * freq0v[idx])
            )
        )
        
        # Relative entropies per position and amino acid:
        Dia[idx] = freq1[idx] * np.log(freq1[idx] / freq0v[idx]) + (
            1 - freq1[idx]
        ) * np.log((1 - freq1[idx]) / (1 - freq0v[idx]))
        
        
    # Overall relative entropies per positions (taking gaps into account):
    Di = np.zeros(N_pos)
    for i in range(N_pos):
        freq1i = freq1[i,:]
        aok = [a for a in range(N_aa) if freq1i[a] > 0]
        flogf = freq1i[aok] * np.log(freq1i[aok] / freqg0[aok])
        Di[i] = flogf.sum()
        freqgi = 1 - freq1i.sum()
        if freqgi > tolerance:
            Di[i] += freqgi * np.log(freqgi / theta)
    return Wia, Dia, Di

def scaMat(alg, seqw=1, norm="none", lbda=0, freq0=np.ones(21) / 21):
    
    N_seq, N_pos = alg.shape
    N_aa = freq0.shape[0]
    
    if isinstance(seqw, int) and seqw == 1:
        seqw = np.ones((1, N_seq))

    #print("SCA Freqs")        
    # use uniform distro here bc that's how it's done in SCA... input
    # freq0 is the default, which is uniform
    freq1, freq2, _ = freq(
        alg, Naa=N_aa, seqw=seqw, lbda=lbda, freq0=np.ones(N_aa)/21
    )
    
    #print("SCA PosWeights") 
    # I want to be able to pass in my modified background freqs
    # which are necessary for posWeights. Previously defaults are used
    Wpos, _, _ = posWeights(alg, seqw, lbda, N_aa, freq0)

    #print("SCA TildeC")
    tildeC = np.zeros((N_pos, N_pos, N_aa, N_aa))
    for i in range(N_pos):
        for j in range(i, N_pos):
            tildeC[i,j] = np.outer(Wpos[i],Wpos[j]) * (freq2[i,j] - np.outer(freq1[i],freq1[j]))
            if i != j:
                tildeC[j,i] = np.outer(Wpos[j], Wpos[i]) * (freq2[j,i] - np.outer(freq1[j], freq1[i]))    

    if norm == 'none':
        return tildeC, None

    # Positional correlations:
    Cspec = np.zeros((N_pos, N_pos))
    Cfrob = np.zeros((N_pos, N_pos))
    for i in range(N_pos):
        for j in range(i, N_pos):
            u, s, vt = np.linalg.svd(
                tildeC[i, j, 1:, 1:]
            )
            Cspec[i, j] = s[0]
            Cfrob[i, j] = np.sqrt(sum(s ** 2))
    Cspec += np.triu(Cspec, 1).T
    Cfrob += np.triu(Cfrob, 1).T

    if norm == "frob":
        Cspec = Cfrob
        
    return tildeC, Cspec

def prune_params(align, outfile_prefix="align", outfile_suffix=".mat",
					cij=True, fij = False, weight=0.7, pct=95):
	"""
	Given a .mat alignment, fasta file, or .npy alignment, create the pruning mask for input to DCA/SBM.

	Parameters:
	align : file with alignment data in either .fasta, .mat, or .npy format
	outfile_prefix : unique identifier for output file (along with pruning type,
						reweighting threshold, and pruning percentage)
	outfile_suffix : file type to save the output file as. Default is .mat, but .npy is
						also supported. No other file types supported
	cij : prune based on statistical coupling analysis (SCA) C_tilde(i,j,a,b).
			See Rivoire, Reynolds, and Ranganathan 2016 for details. Default True.
	fij : prune based on second order alignment statistics. Default False.
	weight : threshold value for sequence re-weighting (if hamming dist < weight, 
				two sequences are "the same")
	pct : percent of parameters to EXCLUDE from the pruning mask, as a percentage of L*L*q*q/2
			(so not quite L choose 2 * q * q; kept as slightly higher number for historical reasons
			at this point). Can be a single number or a list of percentages.
			
	Returns: None, but writes pruning masks to file(s).

	Note: Both cij and fij flags can be set at the same time. This will output two files for
			provided pct values, one for each strategy.
	"""

	assert cij or fij, "Either cij or fij pruning needs to be set (or both)"

	# print("Loading matrix...")
	# alg = ""
	# if align[-4:] == ".mat":
	# 	# load alignment, convert to python indexing
	# 	alg = sio.loadmat(align)['align'] - 1
	# elif align[-6:] == ".fasta":
	# 	alg = ut.load_fasta(align)
	# elif align[-4:] == ".npy":
	# 	alg = np.load(align)
	# else:
	# 	print("Unsupported input file type")
	# 	return
    
	alg = np.copy(align)

	L = alg.shape[1]
	Nseq = alg.shape[0]

	if not isinstance(pct, list):
		pct = [pct]

	if 100 in pct:
		print("Pruning all parameters")
		
		prune_mat = np.zeros((L,L,21,21), dtype='int')
		prune_mat = prune_mat.transpose(2,3,0,1) # swap indices for consistency with MATLAB code

		print("Writing files...")
		# write pruned matrix

		outfile = "%s_%s_%.2fp_SeqW_%.1f%s"%(outfile_prefix, "Fij", 100, weight, outfile_suffix)
		if outfile_suffix == ".npy":
			np.save(outfile, prune_mat)
		else:
			prune_mat = prune_mat.transpose(2,3,0,1) # swap indices for consistency with MATLAB code
			sio.savemat(outfile, {'pruneJ':prune_mat})
		pct.remove(100)

	if len(pct) == 0: return

	print("%d positions to look at, %d sequences"%(L, Nseq))

	# calculate sequence weights
	seqw = seqWeights(["".join([numToAA[x] for x in m]) for m in alg],
								max_seqid=0.7, gaps=1, slow=(Nseq > 60000))

	if cij:
		print("Calculating gap frequency...")
		# recalculate background with gaps (pseudocount 0.03 with uniform prior across AAs)
		lbda = 0.03
		seqwn = seqw/seqw.sum()
		background_gaps = (1-lbda) * np.sum(seqwn * (alg==0).sum(axis=1))/ L + 0.03*(1/21)
		freq0_with_gaps = np.hstack([[background_gaps], (1-background_gaps)*freq0_gapless])

		print("Doing SCA...")
		# do SCA calculations
		prune_vals, _ = scaMat(alg,seqw=seqw,lbda=0.03,freq0=freq0_with_gaps)

		print("Organizing parameters...")
		# select parameters to keep
		triu = np.triu_indices(L, k=1) # ignore diagonal
		prune_vals_triu = np.zeros(prune_vals.shape)
		prune_vals_triu[triu] = prune_vals[triu]

		idx = np.argsort(abs(prune_vals_triu).flatten())[::-1] # descending order

		for el in pct:
			tokeep_idx = int((L**2*21**2/2) * (1-el/100))

			prune_mat = np.zeros(prune_vals.size, dtype='int')
			prune_mat[idx[:tokeep_idx]] = 1
			prune_mat = prune_mat.reshape(L,L,21,21)

			for i in range(L):
				for j in range(i+1, L):
					prune_mat[j,i] = prune_mat[i,j].T

			#print("Writing files...")
			# write pruned matrix
			#prune_type = "Cij"
			#outfile = "%s_%s_%.2fp_SeqW_%.1f%s"%(outfile_prefix, prune_type, el, weight, outfile_suffix)
			return prune_mat 
			# if outfile_suffix == ".npy":
			#     np.save(outfile, prune_mat)
			# else:
			#     prune_mat = prune_mat.transpose(2,3,0,1) # swap indices for consistency with MATLAB code
			#     sio.savemat(outfile, {'pruneJ':prune_mat})

	if fij:
		print("Calculating pairwise frequencies...")
		_, prune_vals, _ = freq(alg, seqw, Naa=21, lbda=0)

		print("Organizing parameters...")
		# select parameters to keep
		triu = np.triu_indices(L, k=1) # ignore diagonal
		prune_vals_triu = np.zeros(prune_vals.shape)
		prune_vals_triu[triu] = prune_vals[triu]

		idx = np.argsort(abs(prune_vals_triu).flatten())[::-1] # descending order

		for el in pct:
			tokeep_idx = int((L**2*21**2/2) * (1-el/100))

			prune_mat = np.zeros(prune_vals.size, dtype='int')
			prune_mat[idx[:tokeep_idx]] = 1
			prune_mat = prune_mat.reshape(L,L,21,21)

			for i in range(L):
				for j in range(i+1, L):
					prune_mat[j,i] = prune_mat[i,j].T

			prune_mat = prune_mat.transpose(2,3,0,1) # swap indices for consistency with MATLAB code

			print("Writing files...")
			# write pruned matrix
			prune_type = "Fij"
			outfile = "%s_%s_%.2fp_SeqW_%.1f%s"%(outfile_prefix, prune_type, el, weight, outfile_suffix)

			if outfile_suffix == ".npy":
				np.save(outfile, prune_mat)
			else:
				prune_mat = prune_mat.transpose(2,3,0,1) # swap indices for consistency with MATLAB code
				sio.savemat(outfile, {'pruneJ':prune_mat})

	return None