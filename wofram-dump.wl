
(* --- Begin: libsCP/Quantum2.m --- *)


(* ::Package:: *)

(* {{{ *) BeginPackage["Quantum2`",{"Carlos`", "Quantum`"}]
exportarSilvestre::usage = "lo que dice. sin argumentos."
Isingterm::usage = "Isingterm[i_,j_,N_]"
IsingChain::usage = "IsingChain[J_,N_]"
Hallvsall::usage = "Hallvsall[J_,N_]"
IsingChainInhom::usage = "IsingChainInhom[J_,Jinhom_,N_]"
sigma::usage = "sigma[i_,qubit_,N_] Operador de Pauli i aplicado al qubit con etiqueta qubit, para un sistema con un total de N qubits"
HK::usage = "HK[N_,bx_,bz_]"
matrixU::usage = "matrixU[bx_,bz_,qubits_,topology_]"
IPRSym::usage = "IPRSym[bx_,wk_,topology_]"
PRSym::usage = "PRSym[bx_,wk_,topology_]"
NumberToBinary::usage = "NumberToBinary[u_,bits_]"
ToBinary::usage = "ToBinary[state_]"
ToBase::usage = "ToBase[list_]"
K::usage = "K[qubits_]"
testsym::usage = "testsym[bx_,qubits_,steps_]"
Extractbyk::usage = "Extractbyk[k_,{values_,vecs_}]]"
IPRbyCohstateSymbetter::usage= "IPRbyCohstateSymbetter[\[Theta]_,\[Phi]_,list_,dim_]"
vecsk::usage= "vecsk[qubits_,k_]"
IPRbyCohstateSym::usage= "IPRbyCohstateSym[\[Theta]_, \[Phi]_, bx_, {values_, vecs_}, 
  topology_] "
ModifiedCoherentState::usage = "ModifiedCoherentState[\[Theta]_, \[Phi]_, qubits_]"
ModifiedCoherentState2::usage = "ModifiedCoherentState2[\[Theta]_, \[Phi]_, qubits_]"
IPRbyStatebetter::usage = "IPRbyStatebetter[stateinput_,list_,vecsk_]"
StateToDirac::usage = "StateToDirac[state_,base_:=2] It shows the vector or density matrix in Dirac notation in qubit representation."
CharlieMeasure::usage = "CharlieMeasure[list_] or CharlieMeasure[list_]"
CharlieMeasureAve::usage = "CharlieMeasureAve[list_]"
CharlieMeasureForShowThings::usage = "CharlieMeasureForShowThings[list_]"
CharlieMeasureAveForShowThings::usage = "CharlieMeasureAveForShowThings[list_,deep_]"
StairCase::usage = "StairCase[x_,eigen_]"
NNS::usage = "NNS[eigen_]"
Unfold::usage = "Unfold[list_]"
PBrody::usage = "PBrody[s_,q_]"
LSI::usage = "LSI[unfoldedlist_,bins_] Level Statistic Indicator or simply Gamma parameter"
G::usage = "Amplitude Damping Quantum Channel G[t_,\[Lambda]_,\[Omega]0_,\[Gamma]_]"
H::usage = "Binary Shannon Entropy H[p_]"
QuantumCapacityDamping::usage = "Quantum Capacity of the quantum damping, It must be specified the parameters of G, QuantumCapacityDamping[t_,\[Lambda]_,\[Gamma]_,\[Omega]0_]"
ClassicalCapacityDamping::usage = "EA Classical Capacity of the quantum damping, It must be specified the parameters of G, ClassicalCapacityDamping[t_,\[Lambda]_,\[Gamma]_,\[Omega]0_]"
StepDecomposition::usage = "StepDecomposition[list_,\[Epsilon]_,elemnts_]"
BasisElement::usage = "BasisElement[i_,j_] Dont worry about this, not yet"
BasisElementOneIndex::usage = "BasisElementOneIndex[i_] Dont worry about this, not yet"
DivisibilityKindOf::usage = "DivisivilityKindOf[\[Lambda]1_,\[Lambda]2_,\[Lambda]3_] The lambdas state for the singular values of the unital channel
up rotations), then this function gives 0 when the channel is not CPTP, 1 if the channel is CPTP, 2 if it is p-divisible, 3 if it is compatible
with CP-divisible dynamics and 4 if the channel can be written as exp(L) with L Lindblad."
EntanglementBreakingQ::usage = "EntanglementBreakingQ[x_,y_,z_] this function checks if the channel is entanglement-breaking 
in the sense of a separable Jamilokowski state."
DivisibilityKindOfGeneral::usage = "DivisibilityKindOfGeneral[channel_], Sure it works at least for channels with diagonal lorentz form in the case of CP-divisibility"
gRHP::usage = "gRHP[list_] Calculation of the Rivas g(t) from fidelity, i. e. from D(t) for dephasing channels."
PositiveDerivatives::usage = "PositiveDerivatives[list_] etc."
maximizer::usage = "maximizer[list_] divides the second column of the list by the maximum value of the original list."
QuantumMapInPauliBasis::usage = "QuantumMapInPauliBasis[channel_] This function constructs the Pauli basis channel representation of one qubit"
QuantumMapInUnitBasis::usage = "QuantumMapInInUnitBasis[channel_] This function constructs the Pauli basis channel representation of one qubit"
FromPauliToUnit::usage = " Takes channels in Pauli basis to unital matrices basis."
FromUnitToPauli::usage = " Oposite of From Unit to Pauli."
UnitalChannelInPauliBasis::usage = "UnitalChannelInPauliBasis[x_,y_,z_], where x,y and z are the weights of the corresponding unitaries, by convexity teh weight of the identity operation is automatically determined "
EigensystemOrdered::usage = "EigensystemOrdered[matrix_], Routine for compute Eigensystem with ordered Eigenvalues, see code for further information, this is useful for several routines which require the same basis ordering for different diagonalizations."
RealMatrixLogarithmComplexCase::usage = "RealMatrixLogarithmComplexCase[matrix,k=0] Real logarithm of a matrix for the complex eigenvalues case, returns Falese if the logarithm doesnt exists."
RealMatrixLogarithmRealCase::usage = "RealMatrixLogarithmComplexCase[matrix,k=0] Real logarithm of a matrix for the real eigenvalues case, returns Falese if the logarithm doesnt exists."
HermiticityPreservingAndCCPOfGenerator::usage= "HermiticityPreservingAndCCPOfGenerator[matrix,upto_Branch] Test if the given channels has a hermiticity preserving and ccp generator, returns False if there is no such generator."
HasHermitianPreservingAndCCPGenerator::usage = "HasHermiticitianPreservingAndCCPOfGenerator[matrix_,upto_Branch] Returns True if the channels has hermitian preserving and ccp generator."
DecompositionOfChannelsInSO31::usage = "Performs decomposition in orthochronus Lorentz group of the matrix, returns False if the decomposition can not be done."
LorentzMatrixQ::usage = "LorentzMatrixQ[matrix_] returns if the given matrix is a valid Lorentz transformation with the signature +---."
ForceSameSignatureForLorentz::usage = "ForceSameSignatureForLorentz[matrix_] Forces or fixes the signature of the given Lorentz transformation."
QuantumMaptoR::usage = "QuantumMaptoR[channel_] Representation of Jamiolokowski state in \[Sigma]iotimes \[Sigma]j basis."
diagonalMatrixQ::usage = "diagonalMatrixQ[matrix_] Works similar to the other matrix tests of mathematica."
PositiveSemidefiniteMatrixCustomQ::usage = "Mathematica test gives strange results, this routine check such test hunred times."
PositiveSemidefiniteMatrixCustom2Q::usage = "Custom check."
PositiveSemidefiniteMatrixCustom3Q::usage = "By eigenvalues of the hermitian part"
DecompositionOfChannelsInSO::usage = "Singular value decomposition but in SO."
HermitianPart::usage = "Hermitian part of a matrix."
AntiHermitianPart::usage = "AntiHermitian part of a matrix."
PositiveSemidefinitePart::ysage = "PositiveSemidefinitePart."
PositiveDefinitePart::usage = "PositiveDefinitePart."
NegativeSemidefinitePart::usage = "NegativeSemidefinitePart."
NegativeDefinitePart::usage = "NegativeDefinitePart."
DecompositionOfChannelsInSO31Diagonal::usage= "The same of DecompositionOfChannelsInSO31 but forcing a diagonal in the spatial block."
AddOne::usage="Direct sum from behind with an identity of dimension one."
InfinitySupressor::usage = "InfinitySupressor[lista_,infinity_] This functions makes infinities finide by a given bound."
PartialDecompositionofChannelInSO::usage = "PartialDecompositionofChannelInSO[matrix_]."
TestViaRankOfCPDIV::usage = "TestViaRankOfCPDIV[matrix_], gives True or False."
KrausRank::usage = "Computes the Kraus rank of qubit channels given in Pauli basis."
WolfEisertCubittCiracMeasureQubitCase::usage = "WolfEisertCubittCiracMeasureQubitCase[channel_] Computes waht the name says, checking up to 10 branches."
HilbertSpaceBasisParametrization::usage = "HilbertSpaceBasisParametrization[n_] A most general parametrization of the basis of C^n, or equivalently an element of SU(n)."
PhasesEliminator::usage = "PhasesEliminator[basis_] Removes global phases of the basis given by HilbertSpaceBasisParametrization."
ListIntegrate::usage= "ListIntegrate[list_]."
ForwardNumericalDifferenceForTimeSeries::usage = "ForwardNumericalDifferenceForTimeSeries[a_,n_], where a is the list and n the used finite difference series order to compute the first derivstive."
RealJordanFormOnePair::usage = "RealJordanFormOnePair[channel_]."
CPNoLRegion::usage = "CPNoLRegion[\[Lambda],\[Tau],region] Gives True if the channel parametrized by vectors \[Lambda] and \[Tau] is CP-divisible but not L-divisible, and if it is inside the chosen region. The variable region=1,2,3,0 states for negative octants of channels with positive determinant.
ie, two negative eigenvalues in each region. For example region=1: \[Lambda]1>0,\[Lambda]2<0,\[Lambda]3<0 etc, while region zero explores the whole space. The
function can be also called as CPNoLRegion[\[Lambda],\[Tau]] and CPNoLRegion[\[Lambda]], where autimatically region=0 and \[Tau]=0 & region=0 respectively."
LRegion::usage = "LRegion[\[Lambda]_,\[Tau]_,region_] Gives True if the channel parametrized by vectors \[Lambda] and \[Tau] is L-divisible, and if it is inside the chosen region. The variable region=1,2,3,0 states for negative octants of channels with positive determinant.
ie, two negative eigenvalues in each region. For example region=1: \[Lambda]1>0,\[Lambda]2<0,\[Lambda]3<0 etc, while region zero explores the whole space. The
function can be also called as LRegion[\[Lambda],\[Tau]] and LRegion[\[Lambda]], where autimatically region=0 and \[Tau]=0 & region=0 respectively."
FactorizedSymbolicStateConstructor::usage = "FactorizedSymbolicStateConstructor[n_] gives a factorized state of n particles."
AbstractSwap::usage = "AbstractSwap[init_,i_,j_] operates over factorized states created by FactorizedSymbolicStateConstructor."
AbstractPartialTrace::usage = "AbstractPartialTrace[init_,toleave_] Partial Traces of linear combinations of states generated by FactorizedSymbolicStateConstructor.
The second argument is just the list of particles you want. Hasta la vista."
QuantumMapInProductPauliBasis::usage = "QuantumMapInProductPauliBasis[channel_,particles_] channel in channel[density_matrix] format."
ReshufflingPauliProductBasis::usage = "ReshufflingPauliProductBasis[mat_] Constructs the generalized matrix R of the channel (coefficients of matrix in Pauli product basis)."
ChoiMatrixFromR::usage = "ChoiMatrixFromR[R_] constructs Choi Matrix from Matrix R."
ChoiJamiStateFromPauliProductBasis::usage = "ChoiJamiStateFromPauliProductBasis[channelmatrix_] Constructs  Choi matrix from a channel written in the Pauli Product basis."
Purify::usage = "Purify[\[Rho]_] Purification of rho with a system of the same dimension. The option random->True can be used for random purifications."
ChoiMatrixQubitParticles::usage = "ChoiMatrixQubitParticles[channel_,particles_]."
QuantumMapInProyectorBasis::usage = "QuantumMapInProyectorBasis[channel_,particles_]."
KrausQubitParticles::usage = "KrausQubitParticles[channel_,particles_]."
KrausQubitParticlesFromPauliProductBasis::usage = "KrausQubitParticlesFromPauliProductBasis[channel_?MatrixQ]."
CPQPauli::usage = "Complete positvity test for Pauli channels, with lambdas as the input, CPQPauli[\[Lambda]_]."
PositiveAndNegativeParts::usage = "PositiveAndNegativeParts[M_] returns the positve and negative parts of M such that M=M^+-M^- where M^{+-} are posive-semidefinite and positive, respectively."
ComplementaryChannelFromKrausList::usage = "ComplementaryChannelFromKrausList[list][\[Rho]], computes the complementary channel accordingly Stinespring dilation theorem. Be aware that the complementary channel in general is rectangular, the dimension of the image is the Kraus rank."
PositiveAndNegativeParts::usage = "PositiveAndNegativeParts[M_]: Gives a list with the positive and negative parts of the hermitian matrix M."
KrausFromChoi::usage = "Follow its name"


Begin["Private`"] 
{shift, clock} = {
  {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
  {{1, 0, 0}, {0, Exp[(2 \[Pi] I)/3], 0}, {0, 0,
    Exp[(2 \[Pi] I)/3]^2}}};
silvester =
  MatrixPower[shift, #1] . MatrixPower[clock, #2] & @@@
   Tuples[{0, 1, 2}, {2}];
silvester = 
  Permute[silvester, FindPermutation[{2, 3, 4, 7, 5, 9, 6, 8}]];
silvester = 
  Permute[silvester, FindPermutation[{2, 3, 4, 7, 5, 9, 6, 8}]];

Isingterm[i_,j_,N_]:=Module[{list},
list=Table[If[k==i||k==j,PauliMatrix[3],IdentityMatrix[2]],{k,0,N-1}];
Apply[KroneckerProduct,list]
];

IsingChain[J_,N_]:=J*Sum[Isingterm[i,i+1,N],{i,0,N-2}]+J*Isingterm[N-1,0,N];

Hallvsall[J_,N_]:=Module[{i,j,H},
H=ConstantArray[0,{2^N,2^N}];
For[i=0,i<N,i++,
For[j=1+i,j<N,j++,
H=H+J*Isingterm[i,j,N];
]
];
H
];

IsingChainInhom[J_,Jinhom_,N_]:=J*Sum[Isingterm[i,i+1,N],{i,1,N-2}]+J*Isingterm[N-1,0,N]+Jinhom*Isingterm[0,1,N];

sigma[i_,qubit_,N_]:=Module[{list},
list=Table[If[k==qubit,PauliMatrix[i],IdentityMatrix[2]],{k,0,N-1}];
Apply[KroneckerProduct,list]
];

HK[N_,bx_,bz_]:=bx Sum[sigma[1,qubit,N],{qubit,0,N-1}]+bz Sum[sigma[3,qubit,N],{qubit,0,N-1}];

(*matrixU[bx_,qubits_,topology_]:=Module[{HKi,HI},
If[topology==4,HKi=HK[qubits,bx,1.4]+sigma[1,0,qubits]\[Delta]bx,HKi=HK[qubits,bx,1.4]];
Switch[topology,1,HI=IsingChain[1.0,qubits],2,HI=Hallvsall[1.0,qubits],3,HI=IsingChainInhom[1.0,1.0+\[Delta]J,qubits],4,HI=IsingChain[1.0,qubits]];
MatrixExp[-1.0*I HKi].MatrixExp[-1.0*I HI]
];*)

matrixU[bx_,bz_,qubits_,topology_]:=Module[{HKi,HI},
If[topology==4,HKi=HK[qubits,bx,bz]+sigma[1,0,qubits]\[Delta]bx,HKi=HK[qubits,bx,bz]];
Switch[topology,1,HI=IsingChain[1.0,qubits],2,HI=Hallvsall[1.0,qubits],3,HI=IsingChainInhom[1.0,1.0+\[Delta]J,qubits],4,HI=IsingChain[1.0,qubits]];
MatrixExp[-1.0*I HKi] . MatrixExp[-1.0*I HI]
];

(*matrixU[bx_,qubits_,topology_,\[Delta]_]:=Module[{HKi,HI},
If[topology==4,HKi=HK[qubits,bx,1.4]+sigma[1,0,qubits]\[Delta],HKi=HK[qubits,bx,1.4]];
Switch[topology,1,HI=IsingChain[1.0,qubits],2,HI=Hallvsall[1.0,qubits],3,HI=IsingChainInhom[1.0,1.0+\[Delta],qubits],4,HI=IsingChain[1.0,qubits]];
MatrixExp[-1.0*I HKi].MatrixExp[-1.0*I HI]
];*)

IPRSym[bx_,wk_,topology_,\[Delta]_]:=Module[{U,list,qubits,U0},
qubits=Log[2,Length[Transpose[wk][[1]]]];
U=matrixU[bx,qubits,topology,\[Delta]];
U0=Dagger[wk] . U . wk;
list=Orthogonalize[Eigenvectors[U0]];
1/Length[list]Total[Abs[list]^4,2]
];

PRSym[bx_,wk_,topology_,\[Delta]_]:=Module[{U,list,qubits,U0},
qubits=Log[2,Length[Transpose[wk][[1]]]];
U=matrixU[bx,qubits,topology,\[Delta]];
U0=Dagger[wk] . U . wk;
list=Orthogonalize[Eigenvectors[U0]];
Total[Table[Total[Abs[list[[index]]]^4]^(-1),{index,1,Length[Transpose[wk]]}]]
];

NumberToBinary[u_,bits_]:=Module[{uu,out},uu=u;Reverse[Table[out=Mod[uu,2];uu=IntegerPart[uu/2];out,{bits}]]];

ToBinary[state_]:=NumberToBinary[Position[state,1][[1]][[1]]-1,Log[2,Length[state]]];

ToBase[list_]:=Module[{sum},
sum=1;
Table[If[list[[i]]==1,sum=sum+2^(Length[list]-i)],{i,Length[list]}];
SparseArray[sum->1,2^Length[list]]//Normal
];

K[qubits_]:=Module[{B},
B=ConstantArray[0,2^(2*qubits)];
Table[If[Mod[i,2^qubits+2]==0,B[[i+1]]=1,B[[i+1]]=0],{i,0,2^(2*qubits)/2-1}];
Table[If[Mod[i,2^qubits+2]==1,B[[i+2^(2*qubits)/2+1]]=1,B[[i+2^(2*qubits)/2+1]]=0],{i,0,2^(2*qubits)/2-1}];
Partition[B,2^qubits]
];

testsym[bx_,qubits_,steps_]:=Module[{A,U,sta,values,vecs,list},
U=matrixU[bx,qubits,1];
A=N[K[qubits]];
{values,vecs}=Eigensystem[A];
vecs=Orthogonalize[vecs];
list=Flatten[Table[sta=MatrixPower[U,steps] . vecs[[i]];Chop[A . sta-values[[i]]sta],{i,1,2^qubits}]];
list=DeleteCases[list,0];
If[Length[list]==0,Print["No hay pex"],Print["Preocupate mi cabron"]]
];

Extractbyk[k_,{values_,vecs_}]:=Module[{pos,dim,logdim},
logdim=Log[2,dim];
dim=Length[vecs[[1]]];
pos=Flatten[Position[Round[Chop[Arg[values]*logdim/(2 Pi)]],k]];
Table[#[[i]],{i,pos}]&[vecs]
];

IPRbyCohstateSym[\[Theta]_, \[Phi]_, bx_, {values_, vecs_}, 
  topology_] := Module[{vecs0, w0, U0, U, sta, qubits, list, dim},
  qubits = Log[2, Length[vecs[[1]]]];
  U = matrixU[bx, qubits, topology];
  vecs0 = Extractbyk[0, {values, Orthogonalize[vecs]}];
  dim = vecs0 // Length;
  w0 = Transpose[vecs0];
  sta = Table[
    Chop[QuantumDotProduct[vecs0[[i]], 
      CoherentState[\[Theta], \[Phi], qubits]]], {i, dim}];
  U0 = Dagger[w0] . U . w0;
  list = Orthogonalize[Eigenvectors[U0]];
  1/dim Total[
    Table[Abs[QuantumDotProduct[list[[i]], sta]]^4, {i, 1, dim}]]
  ];

IPRbyCohstateSymbetter[\[Theta]_,\[Phi]_,list_,vecsk_]:=Module[{state},
state=Table[Chop[QuantumDotProduct[vecsk[[i]],CoherentState[\[Theta],\[Phi],Log[2,Length[vecsk[[1]]]]]]],{i,Length[vecsk]}];
Total[Table[Abs[QuantumDotProduct[list[[i]],state]]^4,{i,1,Length[vecsk]}]]
];

IPRbyStatebetter[stateinput_,list_,vecsk_]:=Module[{state},
state=Table[Chop[QuantumDotProduct[vecsk[[i]],stateinput]],{i,Length[vecsk]}];
Total[Table[Abs[QuantumDotProduct[list[[i]],state]]^4,{i,1,Length[vecsk]}]]
];

vecsk[qubits_,k_]:=Module[{values,vecs},
{values, vecs} = Eigensystem[N[K[qubits]]];
Extractbyk[k, {values, Orthogonalize[vecs]}]
];

ModifiedCoherentState[\[Theta]_, \[Phi]_, qubits_] := 
 Flatten[KroneckerProduct[CoherentState[\[Theta], \[Phi], 3], 
   PauliMatrix[1] . CoherentState[\[Theta], \[Phi], 1], 
   CoherentState[\[Theta], \[Phi], qubits - 4]], 1];

ModifiedCoherentState2[\[Theta]_, \[Phi]_, qubits_] := 
 Flatten[KroneckerProduct[CoherentState[\[Theta], \[Phi], 2], 
   KroneckerProduct[PauliMatrix[1] . CoherentState[\[Theta], \[Phi], 1],
     PauliMatrix[1] . CoherentState[\[Theta], \[Phi], 1]], 
   CoherentState[\[Theta], \[Phi], qubits - 4]], 1];

StateToDirac[state_?VectorQ]:=Sum[state[[i]]*"|"<>(TableForm[{IntegerDigits[i-1,2,Log2[Length[state]]//Round]}, TableSpacing->{1.2,1.2}]//ToString)<>"\[RightAngleBracket]",{i,1,Length[state]}];

StateToDirac[state_?MatrixQ]:=Sum[state[[i,j]]*"|"<>(TableForm[{IntegerDigits[i-1,2,Log2[Length[state]]//Round]}, TableSpacing->{1.2,1.2}]//ToString)<>"\[RightAngleBracket]\[LeftAngleBracket]"<>(TableForm[{IntegerDigits[j-1,2,Log2[Length[state]]//Round]}, TableSpacing->{1.2,1.2}]//ToString)<>"|",{i,1,Length[state]},{j,1,Length[state]}];

StateToDirac[state_?MatrixQ,base_]:=Sum[state[[i,j]]*"|"<>(TableForm[{IntegerDigits[i-1,base,Log[base,Length[state]]//Round]}, TableSpacing->{1.2,1.2}]//ToString)<>"\[RightAngleBracket]\[LeftAngleBracket]"<>(TableForm[{IntegerDigits[j-1,base,Log[base,Length[state]]//Round]}, TableSpacing->{1.2,1.2}]//ToString)<>"|",{i,1,Length[state]},{j,1,Length[state]}];

StateToDirac[state_?VectorQ,base_]:=Sum[state[[i]]*"|"<>(TableForm[{IntegerDigits[i-1,base,Log[base, Length[state]]//Round]}, TableSpacing->{1.2,1.2}]//ToString)<>"\[RightAngleBracket]",{i,1,Length[state]}];

CharlieMeasure[list_] := 
  Module[{l, listD, Criticallistmin, Criticallistmax, position, maxi, 
    len},
   l = Length[list];
   listD = Table[list[[i + 1]] - list[[i]], {i, l - 1}];
   Criticallistmax = 
    Sort[DeleteCases[
      Table[If[(listD[[i]] > 0 && 
           listD[[i + 1]] <= 0) || (listD[[i]] >= 0 && 
           listD[[i + 1]] < 0), {i, list[[i + 1]]}, 0], {i, l - 2}], 
      0], #1[[2]] > #2[[2]] &];
    Criticallistmin=Table[If[(listD[[i]]<=0&&listD[[i+1]]>0)||(listD[[i]]<0&&listD[[i+1]]>=0),list[[i+1]],1],{i,l-2}];
len=Length[Criticallistmax];
If[len==0,0,Max[{0,Max[
Table[Criticallistmax[[i]][[2]]-Min[Take[Criticallistmin,Criticallistmax[[i]][[1]]]],{i,len}]
]}]]
];

CharlieMeasureAve[list_] := 
  Module[{l, listD, Criticallistmin, Criticallistmax, position, maxi, 
    len},
   l = Length[list];
   listD = Table[list[[i + 1]] - list[[i]], {i, l - 1}];
   Criticallistmax = 
    Sort[DeleteCases[
      Table[If[(listD[[i]] > 0 && 
           listD[[i + 1]] <= 0) || (listD[[i]] >= 0 && 
           listD[[i + 1]] < 0), {i, list[[i + 1]]}, 0], {i, l - 2}], 
      0], #1[[2]] > #2[[2]] &];
len=Length[Criticallistmax];
If[len==0,0,Max[{0,Max[Table[Criticallistmax[[i]][[2]]-Mean[Take[list,Criticallistmax[[i]][[1]]-1]],{i,1,len}]]}]]
]

CharlieMeasureForShowThings[list_,deep_]:=Module[{l,listD,Criticallistmin,Criticallistmax,position,maxi,len,tab,positionmax,miningraph,maxingraph,minlist,positionmin},
l=Length[list];
listD=Table[list[[i+1]]-list[[i]],{i,l-1}];
Criticallistmax=Sort[DeleteCases[Table[If[(listD[[i]]>0&&listD[[i+1]]<=0)||(listD[[i]]>=0&&listD[[i+1]]<0),{i,list[[i+1]]},0],{i,l-2}],0],#1[[2]]>#2[[2]]&];
Criticallistmin=Table[If[(listD[[i]]<=0&&listD[[i+1]]>0)||(listD[[i]]<0&&listD[[i+1]]>=0),list[[i+1]],1],{i,l-2}];
len=Length[Criticallistmax];
If[deep>len,Print["No hay tantos maximos"]; Abort[];];
tab=Table[Criticallistmax[[i]][[2]]-Min[Take[Criticallistmin,Criticallistmax[[i]][[1]]]],{i,If[deep==0,len,deep]}];
positionmax=Flatten[Position[tab,maxi=Max[{Max[tab],0}]]]//Last;
maxingraph=Criticallistmax[[positionmax]][[1]];
miningraph=Min[minlist=Take[Criticallistmin,maxingraph]];
positionmin=Flatten[Position[minlist,miningraph]]//Last;
Show[ListLinePlot[list,PlotRange->All,PlotStyle->Red,PlotLabel->Style[ToString[maxi]]],ListPlot[{{positionmin+1,list[[positionmin+1]]},{maxingraph+1,list[[maxingraph+1]]}},PlotStyle->{Blue,PointSize[0.015]}],ImageSize->Medium]
];

CharlieMeasureAveForShowThings[list_,deep_]:=Module[{mean,l,listD,Criticallistmin,Criticallistmax,position,maxi,len,tab,positionmax,miningraph,maxingraph,minlist,positionmin},
l=Length[list];
listD=Table[list[[i+1]]-list[[i]],{i,l-1}];
Criticallistmax=Sort[DeleteCases[Table[If[(listD[[i]]>0&&listD[[i+1]]<=0)||(listD[[i]]>=0&&listD[[i+1]]<0),{i,list[[i+1]]},0],{i,l-2}],0],#1[[2]]>#2[[2]]&];
Criticallistmin=Table[If[(listD[[i]]<=0&&listD[[i+1]]>0)||(listD[[i]]<0&&listD[[i+1]]>=0),list[[i+1]],1],{i,l-2}];
len=Length[Criticallistmax];
If[deep>len,Print["No hay tantos maximos"]; Abort[];];
tab=Table[Criticallistmax[[i]][[2]]-Mean[Take[list,Criticallistmax[[i]][[1]]-1]],{i,If[deep==0,len,deep]}];
positionmax=Flatten[Position[tab,maxi=Max[{Max[tab],0}]]]//Last;
maxingraph=Criticallistmax[[positionmax]][[1]];
mean=Mean[Take[list,Criticallistmax[[positionmax]][[1]]-1]];
Show[ListLinePlot[list,PlotRange->All,PlotStyle->Red,PlotLabel->Style[ToString[maxi]]],ListPlot[{{maxingraph+1,list[[maxingraph+1]]}},PlotStyle->{Blue,PointSize[0.015]}],Plot[mean,{x,0,maxingraph},PlotStyle->Directive[Black,Thick]],ImageSize->Medium]
];

CharlieMeasureAveForShowThings[list_]:=CharlieMeasureAveForShowThings[list,0];

CharlieMeasureForShowThings[list_]:=CharlieMeasureForShowThings[list,0];

StairCase[x_,eigen_]:=Length[Select[eigen,#<x&]];

StepDecomposition[list_,\[Epsilon]_,elemnts_]:=Select[Flatten[Split[Sort[#],Abs[#1-#2]<\[Epsilon]&]&/@list,1],Length[#]>elemnts&];

(* Rutinas de Cristopher *)

(*Nearest Neighbour Spacings (NNS) of a list. Preferably Apply after Unfold*)
NNS[eigen_]:=Table[Abs[#[[i+1]]-#[[i]]],{i,Length[#]-1}]&[Sort[eigen]];

(*Unfolding of a list*)
Unfold[list_]:=Module[{List0,Staircase,StairTable,FitStaircase,x},List0=Chop[Sort[#]]-Min[#]&[list];Staircase[x_]:=Length[Select[List0,#<x&]];StairTable=Table[{x,Staircase[x]},{x,Min[#],Max[#],Mean[NNS[#]]}]&[List0];FitStaircase = Fit[StairTable, {1,x,x^2,x^3,x^4,x^5,x^6,x^7,x^8,x^9},x];
FitStaircase/.x->#&/@List0];

Unfold[list_, grado_] :=
  Block[{List0, Staircase, StairTable, FitStaircase, x},
   List0 = Chop[Sort[#]] - Min[#] &[list];
   Staircase[x_] := Length[Select[List0, # < x &]];
   StairTable =
    Table[{x, Staircase[x]}, {x, Min[#], Max[#], Mean[NNS[#]]}] &[
     List0]; FitStaircase =
    Fit[StairTable, Table[x^n, {n, 0, grado}], x];
   FitStaircase /. x -> # & /@ List0];

(*Brody Distribution as function of s, q=0 is Poisson, q=1 is Wigner*)
PBrody[s_,q_]:=Module[{B=Gamma[(2+q)/(1+q)]^(q+1)},B (1+q) s^q*Exp[-B s^(q+1)]];

(*Parametro de Caos LSI, se toma hasta la primera interseccion entre \
Wigner y de Poisson, ya esta region contiene la informacion sobre la \
degeneracion del sistema*)
LSI[unfoldedlist_,bins_] := (4.63551) (Integrate[PDF[HistogramDistribution[unfoldedlist,bins],s],{s,1.0*10^-16,0.472913}] - 0.16109);
(*End of Rutinas de Cristopher*)

(*Amplitude Damping Channel*)
G[t_,\[Lambda]_,\[Omega]0_,\[Gamma]_]:=1/Sqrt[-2 \[Gamma] \[Lambda]+(\[Lambda]+I \[Omega]0)^2] E^(-(1/2) t (\[Lambda]+I \[Omega]0)) (Sqrt[-2 \[Gamma] \[Lambda]+(\[Lambda]+I \[Omega]0)^2] Cosh[1/2 t Sqrt[-2 \[Gamma] \[Lambda]+(\[Lambda]+I \[Omega]0)^2]]+(\[Lambda]+I \[Omega]0) Sinh[1/2 t Sqrt[-2 \[Gamma] \[Lambda]+(\[Lambda]+I \[Omega]0)^2]]);
H[p_]:=-p Log[2,p]-(1-p)Log[2,1-p];
QuantumCapacityDamping[t_,\[Lambda]_,\[Gamma]_,\[Omega]0_]:=If[Abs[G[t,\[Lambda],\[Omega]0,\[Gamma]]]^2>0.5,FindMaximum[H[Abs[G[t,\[Lambda],\[Omega]0,\[Gamma]]]^2 p]-H[(1-Abs[G[t,\[Lambda],\[Omega]0,\[Gamma]]]^2)p],{p,$MinMachineNumber}]//First,0];
ClassicalCapacityDamping[t_,\[Lambda]_,\[Gamma]_,\[Omega]0_]:=FindMaximum[H[p]+H[Abs[G[t,\[Lambda],\[Omega]0,\[Gamma]]]^2 p]-H[(1-Abs[G[t,\[Lambda],\[Omega]0,\[Gamma]]]^2)p],{p,$MinMachineNumber},MaxIterations->Infinity]//First;
(*AveK[list_]:=(Last[list][[1]]-First[list][[1]])^(-1)(list[[2]][[1]]-list[[1]][[1]])Sum[list[[All,2]][[i]],{i,Length[list]}];*)

(*Routines for check divisibility properties*)
BasisElement[i_,j_]:=Table[If[k==i&&l==j,1,0],{k,2},{l,2}];
BasisElementOneIndex[i_]:=Switch[i,1,BasisElement[1,1],2,BasisElement[1,2],3,BasisElement[2,1],4,BasisElement[2,2]];
w=Table[Tr[Dagger[BasisElementOneIndex[i+1]] . PauliMatrix[j]/Sqrt[2]],{i,0,3},{j,0,3}];\[Omega]=Proyector[Bell[2]];\[Omega]ort=IdentityMatrix[4]-\[Omega];

DivisibilityKindOf[\[Lambda]1_,\[Lambda]2_,\[Lambda]3_]:=Module[{eigen,list},
list=Sort[Sqrt[{1,\[Lambda]1^2,\[Lambda]2^2,\[Lambda]3^2}]];
If[
(*Checking Complete Positivity*)
Chop[1+\[Lambda]1+\[Lambda]2+\[Lambda]3]>=0&&Chop[1-\[Lambda]1-\[Lambda]2+\[Lambda]3]>=0&&Chop[1-\[Lambda]1+\[Lambda]2-\[Lambda]3]>=0&&Chop[1+\[Lambda]1-\[Lambda]2-\[Lambda]3]>=0,
If[
(*Evaluating CP-divisibility and p-divisibility*)
(*Evaluating for p-divsibility*)\[Lambda]1 \[Lambda]2 \[Lambda]3>=0,
If[ (*Evaluating for CP-div*)
list[[1]]^2>=\[Lambda]1*\[Lambda]2*\[Lambda]3,
(*Evaluating for markov type evolution*)
If[(*checking hermiticity preserving and CCP*)
(\[Lambda]1*\[Lambda]2*\[Lambda]3>0&&\[Lambda]1/(\[Lambda]2 \[Lambda]3)>=0&&\[Lambda]2/(\[Lambda]1 \[Lambda]3)>=0&&\[Lambda]3/(\[Lambda]2 \[Lambda]1)>=0)||(\[Lambda]1==0&&\[Lambda]2==0&&\[Lambda]3==0)||(\[Lambda]1>0&&\[Lambda]2==0&&\[Lambda]3==0)||(\[Lambda]1==0&&\[Lambda]2>0&&\[Lambda]3==0)||(\[Lambda]1==0&&\[Lambda]2==0&&\[Lambda]3>0)
,4,3
],2],1],0]*0.25];

DivisibilityKindOf[\[Lambda]_]:=DivisibilityKindOf[\[Lambda][[1]],\[Lambda][[2]],\[Lambda][[3]]];

PositiveSemidefiniteMatrixCustomQ[matrix_]:=And@@Table[PositiveSemidefiniteMatrixQ[matrix],{1000}];

PositiveSemidefiniteMatrixCustom2Q[matrix_]:=And@@Table[v=RandomState[Length[matrix]];Chop[QuantumDotProduct[v,HermitianPart[matrix] . v]]>=0,{1000}];

PositiveSemidefiniteMatrixCustom3Q[matrix_]:=And@@NonNegative[Chop[Eigenvalues[HermitianPart[matrix]]]];

g=DiagonalMatrix[{1,-1,-1,-1}];

TestViaRankOfCPDIV[matrix_]:=Module[{c,rank,var,det,list},
c=matrix . g . Transpose[matrix] . g//Chop;
rank=MatrixRank[matrix];
det=Det[matrix];
list=Sqrt[Abs[Eigenvalues[c]]];
If[((var=DiagonalizableMatrixQ[matrix])||(MatrixRank[c]<rank))&&det>=0,
If[var&&(rank<2||(Chop[Apply[Times,Sort[list^2][[{1,4}]]]]>=(Times@@list)&&det>0)),True,
If[(MatrixRank[c]<rank),True,False
],False]
,False]
];

DivisibilityKindOfGeneral[channel_,branches_:1]:=Module[{eigen,list,tmp,det,form},
If[
(*Checking Complete Positivity*)
PositiveSemidefiniteMatrixCustom3Q[tmp=Reshuffle[Chop[FromPauliToUnit[Chop[channel]]]]],
If[
HasHermitianPreservingAndCCPGenerator[Chop[channel],branches],4,
If[TestViaRankOfCPDIV[channel],3,
If[Chop[Det[channel]]>=0,2,1]]
],0]*0.25
];

CPQPauli[\[Lambda]_]:=1/2 (1+\[Lambda][[1]]-\[Lambda][[2]]-\[Lambda][[3]])>=0&&1/2 (1-\[Lambda][[1]]+\[Lambda][[2]]-\[Lambda][[3]])>=0&&1/2 (1-\[Lambda][[1]]-\[Lambda][[2]]+\[Lambda][[3]])>=0&&1/2 (1+\[Lambda][[1]]+\[Lambda][[2]]+\[Lambda][[3]])>=0;

EntanglementBreakingQ[x_,y_,z_]:=If[DivisibilityKindOf[x,y,z]>0,If[Max[0,1/4 (-Abs[-1+x+y-z]-Abs[-1+x-y+z]-Abs[-1-x+y+z]-Abs[1+x+y+z]+8 Max[1/4 Abs[-1+x+y-z],1/4 Abs[-1+x-y+z],1/4 Abs[-1-x+y+z],1/4 Abs[1+x+y+z]])]<=0,2,1],0];

EntanglementBreakingQ[channel_]:=If[Concurrence[0.5*Reshuffle[FromPauliToUnit[channel]]]==0,True,False];

gRHP[list_]:=Map[If[#<0,0,#]&,Table[list[[i+1]]/list[[i]]-1,{i,Length[list]-1}]];
PositiveDerivatives[list_]:=Map[If[#<0,0,#]&,Table[list[[i+1]]-list[[i]],{i,Length[list]-1}]];
(*Misc*)

maximizer[list_,factor_]:=Module[{max},
max=Max[list[[All,2]]];
Map[{#[[1]],#[[2]]/(factor*max)}&,list]
];

maximizer[list_,factor_,deep_]:=Module[{max,maxlist,pos,listnew},
maxlist=Sort[list[[All,2]],Greater];
pos=Table[Position[list[[All,2]],maxlist[[i]]],{i,deep}]//Transpose//First;
listnew=Delete[list,pos];
max=Max[listnew[[All,2]]];
Map[{#[[1]],#[[2]]/(factor*max)}&,listnew]
]

maximizer[list_,factor_,0]:=maximizer[list,factor];

maximizer[list_]:=maximizer[list,1,0];

QuantumMapInPauliBasis[channel_]:=1/2Table[Tr[PauliMatrix[i] . channel[PauliMatrix[j]]],{i,0,3},{j,0,3}];

QuantumMapInUnitBasis[channel_]:=Table[Tr[Dagger[BasisElementOneIndex[i]] . channel[BasisElementOneIndex[j]]],{i,1,4},{j,1,4}];

FromPauliToUnit[channel_]:=w . channel . Dagger[w];
FromUnitToPauli[channel_]:=Dagger[w] . channel . w;

EigensystemOrdered[A_]:=Module[{eigs,vecs,list1,list2},
{eigs,vecs}=Eigensystem[A];
list1=Partition[Riffle[eigs,vecs],2];
list2=Sort[Sort[list1,Re[#1[[1]]]<Re[#2[[1]]]&],Im[#1[[1]]]>Im[#2[[1]]]&];
list2//Transpose
];

RealMatrixLogarithmRealCase[matrix_,k_:0]:=Module[{eig,eigneg,vectors,V,L,pos},
{eig,vectors}=Eigensystem[matrix];
If[Element[eig,Reals]==False,Print["Se uso la rutina para logaritmo complejo"];RealMatrixLogarithmComplexCase[matrix,k],
V=Transpose[vectors]//Chop;
eigneg=Select[eig,#<0&]//Chop;
L=DiagonalMatrix[Log[Abs[eig]]]//Chop;
If[Length[eigneg]==0,V . L . Inverse[V],
If[Length[eigneg]==2&&Chop[eigneg[[1]]-eigneg[[2]],10^(-10)]==0,
pos=Position[eig,eigneg[[1]]][[{1,2},1]];
L[[pos[[1]],pos[[2]]]]=(2 k+1) Pi;
L[[pos[[2]],pos[[1]]]]=-(2 k+1) Pi;
V . L . Inverse[V]//Chop
,False]]
]
];

RealMatrixLogarithmComplexCase[channel_,k_:0]:=Module[{mat,diag,w,e,pos,list,d2,w2,chreorlog,wreal,list2,is},
{diag,w}=Chop[EigensystemOrdered[channel]]//Chop;
If[Element[diag,Reals],(*Print["Se uso la rutina para logaritmo real"];*)RealMatrixLogarithmRealCase[channel,k],
mat=DiagonalMatrix[diag];
w=Transpose[w];
e=ConstantArray[0.0,Dimensions[channel]];
list=Select[diag,Chop[Im[#]]!=0&]//Chop;
list2=Select[diag,Chop[Im[#]]==0&]//Chop;
If[(Length[list2]==2&&AllTrue[list2,NonNegative])==False||(Length[list2]==1)==True,is=False,is=True];
pos=Flatten[Table[Position[diag,i],{i,list}]];
mat[[pos[[1]],pos[[1]]]]=Re[diag[[pos[[1]]]]];
mat[[pos[[2]],pos[[2]]]]=Re[diag[[pos[[1]]]]];
mat[[pos[[1]],pos[[2]]]]=Im[diag[[pos[[1]]]]];
mat[[pos[[2]],pos[[1]]]]=-Im[diag[[pos[[1]]]]];
{d2,w2}=EigensystemOrdered[mat]//Chop;
e[[pos[[1]],pos[[2]]]]=1;
e[[pos[[2]],pos[[1]]]]=-1;
chreorlog=Chop[MatrixLog[mat]+2 Pi k e];
If[Total[Flatten[Chop[d2-diag]]]==0,

w2=Transpose[w2]//Chop;
d2=DiagonalMatrix[d2]//Chop;
wreal=w . Inverse[w2]//Chop;
If[is,wreal . chreorlog . Inverse[wreal]//Chop,is]

,Return["bad calculation"]
]
]
];


HermiticityPreservingAndCCPOfGenerator[matrix_,upto_:1]:=Module[{is,L,i,branches,b},
b=0;
branches=Table[b=b+(-1)^(j+1)*j,{j,0,2*upto}];
is=False;
If[DiagonalizableMatrixQ[matrix],
Table[If[PositiveSemidefiniteMatrixCustom3Q[Chop[\[Omega]ort . Chop[Reshuffle[FromPauliToUnit[L=RealMatrixLogarithmComplexCase[Chop[matrix],k]//Chop]]] . \[Omega]ort]],is=True;i=k;Return[Null,Table],is=False;],{k,branches}];,
Return["non diagonalizable"];
];
If[i!=0,Print["El logaritmo es real hasta k= "<>ToString[i]]];
If[is==True,L,False]
];

DecompositionOfChannelsInSO[map_]:=Module[{a,e,i,newmap,eig,vecs},
{a,e,i}=SingularValueDecomposition[map];
{eig,vecs}=Chop[Eigensystem[Det[a]*Det[i]*(i . e . Transpose[i])]];
{Det[a]*Det[i]*a . Transpose[i] . Transpose[vecs],DiagonalMatrix[eig],vecs}
];

ForceSameSignatureForLorentz[matrix_]:=Module[{tmpmat,eigs,o,tildeM},
tmpmat=matrix . g . Transpose[matrix]//Chop;
{eigs,o}=EigensystemOrdered[tmpmat];tildeM=DiagonalMatrix[eigs];
{o . matrix,o}
];

HermitianPart[m_]:=1/2*(m+Dagger[m]);

AntiHermitianPart[m_]:=1/2*(m-Dagger[m]);

PositiveSemidefinitePart[m_]:=Module[{eig,dec},
eig=Transpose[Eigensystem[m]];
dec=Select[eig,#[[1]]>=0&];
If[Length[dec]==0,0*m,Total[Map[#[[1]]*Proyector[Normalize[#[[2]]]]&,dec]]]
];
PositiveDefinitePart[m_]:=Module[{eig,dec},
eig=Transpose[Eigensystem[m]];
dec=Select[eig,#[[1]]>0&];
If[Length[dec]==0,0*m,Total[Map[#[[1]]*Proyector[Normalize[#[[2]]]]&,dec]]]
];
NegativeSemidefinitePart[m_]:=Module[{eig,dec},
eig=Transpose[Eigensystem[m]];
dec=Select[eig,#[[1]]<=0&];
If[Length[dec]==0,0*m,-Total[Map[#[[1]]*Proyector[Normalize[#[[2]]]]&,dec]]]
];
NegativeDefinitePart[m_]:=Module[{eig,dec},
eig=Transpose[Eigensystem[m]];
dec=Select[eig,#[[1]]<0&];
If[Length[dec]==0,0*m,-Total[Map[#[[1]]*Proyector[Normalize[#[[2]]]]&,dec]]]
];

QuantumMaptoR[channel_]:=Module[{g,\[Rho]},
g=DiagonalMatrix[{1,-1,-1,-1}];
\[Rho]=Reshuffle[FromPauliToUnit[channel]]/2//Chop;
(Table[Tr[\[Rho] . KroneckerProduct[PauliMatrix[i],PauliMatrix[j]]],{i,0,3},{j,0,3}])//FullSimplify//Chop
];

LorentzMatrixQ[matrix_]:=
Chop[matrix[[1,1]]]>0&&Chop[Det[matrix]]==1&&Chop[matrix . g . Transpose[matrix]]==g;

DecompositionOfChannelsInSO31[matrix_]:=Module[{c,x,j,n,eig1,eig2,o,d,leftL,rightL},
c=g . matrix . g . Transpose[matrix]//Chop;
{x,j}=SchurDecomposition[c]//Chop;x=Inverse[x]//Chop;
n=x . g . Transpose[x]//Chop;
{eig1,o}=EigensystemOrdered[n]//Chop;
leftL=Transpose[x] . Transpose[o]//Chop;
c=g . Transpose[matrix] . g . matrix;
{x,j}=SchurDecomposition[c]//Chop;x=Inverse[x]//Chop;
n=x . g . Transpose[x]//Chop;
{eig2,o}=EigensystemOrdered[n]//Chop;
If[Chop[eig1]==Diagonal[g]&&eig2==Diagonal[g],None,Print["Wrong calculation"];Abort[];];
rightL=Transpose[x] . Transpose[o]//Chop;
If[Det[rightL]==-1,rightL=Sign[rightL[[1,1]]]g . rightL;,If[rightL[[1,1]]<0,rightL=-g . g . rightL;]];
If[Det[leftL]==-1,leftL=Sign[leftL[[1,1]]]g . leftL;,If[leftL[[1,1]]<0,leftL=-g . g . leftL;]];
If[(LorentzMatrixQ[leftL]&&LorentzMatrixQ[rightL])==False,Print["Decomposition not done"];,{leftL,Transpose[leftL] . matrix . rightL,rightL}//Chop]
];

AddOne[matrix_]:=Flatten[{{{1.0,0.0,0.0,0.0}},Table[Flatten[{0.0,matrix[[i,All]]}],{i,1,3}]},1];

DecompositionOfChannelsInSO31Diagonal[matrix_]:=Module[{c,x,j,n,eig1,eig2,o,d,leftL,rightL,form,aux,form2,a,e,i},
(*Transformada Izquierda:*)
c=g . matrix . g . Transpose[matrix]//Chop;
{x,j}=SchurDecomposition[c]//Chop;x=Inverse[x]//Chop;
n=x . g . Transpose[x]//Chop;
{eig1,o}=EigensystemOrdered[n]//Chop;
leftL=Transpose[x] . Transpose[o]//Chop;
(*Transformada Derecha:*)
c=g . Transpose[matrix] . g . matrix;
{x,j}=SchurDecomposition[c]//Chop;x=Inverse[x]//Chop;
n=x . g . Transpose[x]//Chop;
{eig2,o}=EigensystemOrdered[n]//Chop;
If[Chop[eig1]==Diagonal[g]&&eig2==Diagonal[g],None,Print["Wrong calculation"];Abort[];];
rightL=Transpose[x] . Transpose[o]//Chop;
(*Se define la forma normal preliminar, pues puede que no salga diagonal*)
form=Transpose[leftL] . matrix . rightL;
e=form;
(*En este paso construyo un canal solo con el bloque espacial de la matriz: *)
form2=AddOne[Take[form,{2,4},{2,4}]];
(*Por si no sale diagonal el bloque espacial, aqui se fuerza descomponiendolo en SO(3):*)
If[DiagonalMatrixQ[form2]==False,
{a,e,i}=DecompositionOfChannelsInSO[form2]//Chop;
aux=form-form2;
d=aux[[All,1]];
d=Transpose[a] . d//Chop;
e[[All,1]]=d;e[[1,1]]=1;
(*Se redefinen las transformaciones de Lorentz con las nuevas rotaciones:*)
leftL=leftL . a//Chop;
rightL=rightL . Transpose[i]//Chop;
];
(*Se hacen propias y ortocronas usando el grupo cociente O(3,1)/SO^+(3,1):*)
If[Det[rightL]==-1,rightL=Sign[rightL[[1,1]]]g . rightL;,If[rightL[[1,1]]<0,rightL=-g . g . rightL;]];
If[Det[leftL]==-1,leftL=Sign[leftL[[1,1]]]g . leftL;,If[leftL[[1,1]]<0,leftL=-g . g . leftL;]];
(*Un test util para mantener esta funcion y el output:*)
If[(LorentzMatrixQ[leftL]&&LorentzMatrixQ[rightL])==False,Print["Decomposition not done"];,{leftL,e,rightL}//Chop]
];

PartialDecompositionofChannelInSO[matrix_]:=Module[{form2,a,e,i,aux,form,d,leftL,rightL},
form2=AddOne[Take[matrix,{2,4},{2,4}]];
{a,e,i}=DecompositionOfChannelsInSO[form2]//Chop;
aux=matrix-form2;
d=aux[[All,1]];
d=Transpose[a] . d//Chop;
e[[All,1]]=d;e[[1,1]]=1;
leftL=a//Chop;
rightL=Transpose[i]//Chop;
{leftL,e,rightL}
];

HasHermitianPreservingAndCCPGenerator[matrix_,upto_:1]:=Module[{is,L,i,branches,b},
b=0;
branches=Table[b=b+(-1)^(j+1)*j,{j,0,2*upto}];
is=False;
If[DiagonalizableMatrixQ[matrix],

Table[If[PositiveSemidefiniteMatrixCustom3Q[\[Omega]ort . FullSimplify[Reshuffle[FromPauliToUnit[L=RealMatrixLogarithmComplexCase[Chop[matrix],k]//Chop]//Chop]] . \[Omega]ort//Chop],is=True;i=k;Return[Null,Table],is=False;],{k,branches}];,
Return["non diagonalizable"];
];
If[i!=0,Print["Hermiticity preserving and ccp condition is fulfilled until k= "<>ToString[i]]];
If[is==True,is,False]
];

WolfEisertCubittCiracMeasureQubitCase[channel_,OptionsPattern[{branches->5,noisedelta->0.01}]]:=Module[{dev,noise,lol,\[Mu]},
noise[\[Mu]_]:=-\[Mu] DiagonalMatrix[{0,1,1,1}];
dev=RealMatrixLogarithmComplexCase[channel,10];
If[Length[dev]==0,0,
lol=Min[DeleteCases[Table[If[PositiveSemidefiniteMatrixCustom3Q[Chop[\[Omega]ort . Reshuffle[FromPauliToUnit[RealMatrixLogarithmComplexCase[channel,i]+noise[j]]] . \[Omega]ort]],j,None],{i,0,OptionValue[branches]},{j,0.0,1.0,OptionValue[noisedelta]}]//Flatten,None]];
Exp[-3lol]
]
];

UnitalChannelInPauliBasis[x_,y_,z_]:=DiagonalMatrix[(1-x-y-z){1,1,1,1}+x{1,1,-1,-1}+y{1,-1,1,-1}+z{1,-1,-1,1}];
UnitalChannelInPauliBasis[vec_]:=UnitalChannelInPauliBasis[vec[[1]],vec[[2]],vec[[3]]];

diagonalMatrixQ[matrix_]:=If[Total[Abs[Flatten[DiagonalMatrix[Diagonal[matrix]]-matrix]]]==0,True,False];

InfinitySupressor[lista_,infinity_]:=Module[{list},
list=lista;
Table[If[list[[i]][[2]]>infinity,list[[i]][[2]]=infinity],{i,Length[list]}];
list
];

KrausRank[channel_]:=MatrixRank[Reshuffle[FromPauliToUnit[channel]]];

ListIntegrate[list_]:=Module[{sum,step},
step=list[[2]][[1]]-list[[1]][[1]];
sum=0.0;
step*Table[sum=sum+list[[i]],{i,1,Length[list]}]
];

DeltaVector[a_,size_]:=Module[{l},
l=ConstantArray[0,size];
Table[l[[i]]=a[[i+1]]-a[[i]],{i,Length[a]-1}];
l
];

ForwardNumericalDifferenceForTimeSeries[a_,n_]:=Module[{x},Take[Sum[SeriesCoefficient[Log[1+x],{x,0,i}]*(Nest[DeltaVector[#,Length[a]]&,a,i]),{i,1,n}],Length[a]-n]];
	
CPNoLRegion[\[Lambda]_,\[Tau]_,region_]:=Module[{cha,regcheck,vals,cptp},
cha:={{1,0,0,0},{\[Tau][[1]],\[Lambda][[1]],0,0},{\[Tau][[2]],0,\[Lambda][[2]],0},{\[Tau][[3]],0,0,\[Lambda][[3]]}};
regcheck=Switch[region,1,\[Lambda][[1]]>=0&&\[Lambda][[2]]<=0&&\[Lambda][[3]]<=0,2,\[Lambda][[1]]<=0&&\[Lambda][[2]]>=0&&\[Lambda][[3]]<=0,3,\[Lambda][[1]]<=0&&\[Lambda][[2]]<=0&&\[Lambda][[3]]>=0,0,True];
cptp=And@@Thread[(cha//FromPauliToUnit//Reshuffle//Eigenvalues//N//Chop)>0];
vals=cha . g . Transpose[cha] . g//Eigenvalues//Abs//Sqrt;
(Min[vals]^2*Max[vals]^2>=Times@@vals)&&(Det[cha]>0)&&Not[\[Lambda][[1]]>=0&&\[Lambda][[2]]>=0&&\[Lambda][[3]]>=0]&&regcheck&&cptp
];
CPNoLRegion[\[Lambda]_,\[Tau]_]:=CPNoLRegion[\[Lambda],\[Tau],0];
CPNoLRegion[\[Lambda]_]:=CPNoLRegion[\[Lambda],{0,0,0},0];

LRegion[\[Lambda]_,\[Tau]_,region_]:=Module[{cha,regcheck,cptp},
cha:={{1,0,0,0},{\[Tau][[1]],\[Lambda][[1]],0,0},{\[Tau][[2]],0,\[Lambda][[2]],0},{\[Tau][[3]],0,0,\[Lambda][[3]]}};regcheck=Switch[region,1,\[Lambda][[1]]>=0&&\[Lambda][[2]]<=0&&\[Lambda][[3]]<=0,2,\[Lambda][[1]]<=0&&\[Lambda][[2]]>=0&&\[Lambda][[3]]<=0,3,\[Lambda][[1]]<=0&&\[Lambda][[2]]<=0&&\[Lambda][[3]]>=0,0,True];cptp=And@@Thread[(cha//FromPauliToUnit//Reshuffle//Eigenvalues//N//Chop)>0];
If[DiagonalizableMatrixQ[cha//Chop],HasHermitianPreservingAndCCPGenerator[Chop[cha],1]&&regcheck,False]
];
LRegion[\[Lambda]_,\[Tau]_]:=LRegion[\[Lambda],\[Tau],0];
LRegion[\[Lambda]_]:=LRegion[\[Lambda],{0,0,0},0];

RealJordanFormOnePair[channel_]:=Module[{mat,diag,w,e,pos,list,d2,w2,chreorlog,wreal,list2,is},
{diag,w}=Chop[EigensystemOrdered[channel]]//Chop;
If[Element[diag,Reals],JordanDecomposition[channel],
mat=DiagonalMatrix[diag];
w=Transpose[w];
e=ConstantArray[0.0,Dimensions[channel]];
list=Select[diag,Chop[Im[#]]!=0&]//Chop;
list2=Select[diag,Chop[Im[#]]==0&]//Chop;
If[(Length[list2]==2&&AllTrue[list2,NonNegative])==False||(Length[list2]==1)==True,is=False,is=True];
pos=Flatten[Table[Position[diag,i],{i,list}]];
mat[[pos[[1]],pos[[1]]]]=Re[diag[[pos[[1]]]]];
mat[[pos[[2]],pos[[2]]]]=Re[diag[[pos[[1]]]]];
mat[[pos[[1]],pos[[2]]]]=Im[diag[[pos[[1]]]]];
mat[[pos[[2]],pos[[1]]]]=-Im[diag[[pos[[1]]]]];
{d2,w2}=EigensystemOrdered[mat]//Chop;
If[Total[Flatten[Chop[d2-diag]]]==0,

w2=Transpose[w2]//Chop;
d2=DiagonalMatrix[d2]//Chop;
wreal=w . Inverse[w2]//Chop;
If[is,{wreal,mat//Chop},is]
,Return["bad calculation"]
]
]];

(*Symbolic tools for factorized mixed states*)

PreAbstractSwap[init_,i_,j_]:=Module[{aux,aux2},
If[ToString[Head[init//Expand]]=="Times",
len=Length[init];
aux=Last[init];
aux2=aux;
aux2[[i]]=aux[[j]];
aux2[[j]]=aux[[i]];
Take[init,{1,len-1}]*aux2,
If[ToString[Head[init]]=="TensorProduct",
aux2=init;
aux2[[i]]=init[[j]];
aux2[[j]]=init[[i]];
aux2
]
]
];
AbstractSwap[init_,i_,j_]:=If[ToString[Head[init//Expand]]=="Plus",Map[PreAbstractSwap[#,i,j]&,init//Expand],PreAbstractSwap[init//Expand,i,j]];
PreAbstractPartialTrace[init_,parttoleave_]:=Module[{len,aux,new},
If[ToString[Head[init//Expand]]=="Times",
len=Length[init];
aux=Last[init];
new=aux[[parttoleave]];
Take[init,{1,len-1}]*new,
If[ToString[Head[init]]=="TensorProduct",
init[[parttoleave]]
]
]
];
AbstractPartialTrace[init_,toleave_]:=If[ToString[Head[init//Expand]]=="Plus",Map[PreAbstractPartialTrace[#,toleave]&,init//Expand],PreAbstractPartialTrace[init//Expand,toleave]];
	
	
(*Channel operations in product Pauli basis*)

QuantumMapInProductPauliBasis[channel_,particles_]:=Module[{},
Table[1/2^particles Tr[Pauli[IntegerDigits[i-1,4,particles]] . channel[Pauli[IntegerDigits[j-1,4,particles]]]],{i,1,2^(2particles)},{j,1,2^(2particles)}]//Chop
];

ReshufflingPauliProductBasis[mat_]:=Module[{particles,columns,aux},
particles=Log[2,Length[mat]]/2;
aux= mat;
columns=Position[Table[(-1)^Count[IntegerDigits[i-1,4,particles],2],{i,1,2^(2particles)}],-1]//Flatten;
Table[aux[[All,i]]=-aux[[All,i]];,{i,columns}];
aux
];

ChoiMatrixFromR[R_]:=Module[{particles},
particles=Log[2,Length[R]]/2;
Sum[1/2^(2 particles) R[[i,j]]*Pauli[Flatten[{IntegerDigits[i-1,4,particles],IntegerDigits[j-1,4,particles]}]],{i,1,2^(2particles)},{j,1,2^(2particles)}]
];

ChoiJamiStateFromPauliProductBasis[channelmatrix_]:=ChoiMatrixFromR[ReshufflingPauliProductBasis[channelmatrix]];

(*Purification routines*)

Purify[\[Rho]_,OptionsPattern[{random->False}]]:=Module[{vals,vecs,U},
{vals,vecs}=Eigensystem[\[Rho]];
vecs=Map[Normalize,Orthogonalize[vecs]];
If[OptionValue[random],U=CUEMember[Length[\[Rho]]];,U=IdentityMatrix[Length[\[Rho]]]];
Sum[Sqrt[vals[[i]]]Flatten[KroneckerProduct[vecs[[i]],U . vecs[[i]]]],{i,1,Dimensions[\[Rho]][[1]]}]
]

ChoiMatrixQubitParticles[channel_,particles_]:=ArrayFlatten[Table[channel[Proyector[BasisState[i,2^particles],BasisState[j,2^particles]]],{i,0,2^particles-1},{j,0,2^particles-1}]];

QuantumMapInProyectorBasis[channel_,particles_]:=Flatten[Table[Flatten[Table[Tr[Dagger[Proyector[BasisState[k,2^particles],BasisState[l,2^particles]]] . channel[Proyector[BasisState[i,2^particles],BasisState[j,2^particles]]]],{i,0,2^particles-1},{j,0,2^particles-1}]],{k,0,2^particles-1},{l,0,2^particles-1}],1];

KrausFromChoi[choi_]:=Select[
Map[Sqrt[#[[1]]]Partition[Normalize[#[[2]]],Sqrt[Length[choi]]]&,choi//Eigensystem//Transpose]//Chop,MatrixRank[#]>0 &];

KrausQubitParticles[channel_,particles_]:=KrausFromChoi[ChoiMatrixQubitParticles[channel,particles]];

KrausQubitParticlesFromPauliProductBasis[channel_?MatrixQ]:=KrausFromChoi[Round[Sqrt[Length[channel]]]*ChoiJamiStateFromPauliProductBasis[channel]]

(*XX-chain section*)

PositiveAndNegativeParts[M_]:=Module[{eig,PositiveM,NegativeM},
eig=Eigensystem[M];
Select[eig//Chop//Transpose,#[[1]]>= 0&];
PositiveM=Select[eig//Chop//Transpose,#[[1]]>= 0&];
NegativeM=Select[eig//Chop//Transpose,#[[1]]< 0&];
{Sum[PositiveM[[i]][[1]]Proyector[PositiveM[[i]][[2]]],{i,Length[PositiveM]}],Sum[-NegativeM[[i]][[1]]Proyector[NegativeM[[i]][[2]]],{i,Length[NegativeM]}]}
];

ComplementaryChannelFromKrausList[list_][\[Rho]_]:=Module[{dim},
dim=Length[list];
Sum[Tr[list[[i]] . \[Rho] . Dagger[list[[j]]]]Proyector[BasisState[i-1,dim],BasisState[j-1,dim]],{i,dim},{j,dim}]
];

PositiveAndNegativeParts[M_]:=Module[{eig,PositiveM,NegativeM},
eig=Eigensystem[M];
Select[eig//Chop//Transpose,#[[1]]>= 0&];
PositiveM=Select[eig//Chop//Transpose,#[[1]]>= 0&];
NegativeM=Select[eig//Chop//Transpose,#[[1]]< 0&];
{Sum[PositiveM[[i]][[1]]Proyector[PositiveM[[i]][[2]]],{i,Length[PositiveM]}],Sum[-NegativeM[[i]][[1]]Proyector[NegativeM[[i]][[2]]],{i,Length[NegativeM]}]}
];

End[]

FactorizedSymbolicStateConstructor[dim_]:=TensorProduct@@Table[Subscript[\[Rho], i],{i,1,dim,1}];

SinCosList[j_,vec_]:=Table[If[i==j,Cos,Sin][\[Theta][vec][i+1]],{i,0,j}];	
	
SinList[j_,vec_]:=Table[Sin[\[Theta][vec][i+1]],{i,0,j}];

SinTerm[j_,vec_]:=Times@@SinList[j,vec];

SinCosTerm[j_,vec_]:=Times@@SinCosList[j,vec];

Vector[index_,dim_,basis_]:=Sum[If[dim-1==i,SinTerm[i-1,index],SinCosTerm[i,index]]Exp[I \[Phi][index][i]]basis[[i+1]],{i,0,dim-1}];

Vector[index_,dim_]:=Table[If[dim-1==i,SinTerm[i-1,index],SinCosTerm[i,index]]Exp[I \[Phi][index][i]],{i,0,dim-1}];

vecuptoU[vec_,index_,b_]:=ReplaceAll[D[vec,\[Theta][index][b]],Table[\[Theta][index][i]->Pi/2,{i,1,b-1}]];

BasisComplement[vec_,index_,dim_]:=Table[vecuptoU[vec,index,i],{i,dim-(index+1)}];

HilbertSpaceBasisParametrization[n_]:=Module[{vec,basis},
vec=Vector[0,n];
Flatten[{{vec},Table[
basis=BasisComplement[vec,i-1,n];
vec=Vector[i,n-i,basis];
vec
,{i,n-1}]},1]
];

PhasesEliminator[basis_]:=Module[{len},
len=Length[basis];
ReplaceAll[basis,Table[\[Phi][i][0]->0,{i,0,len-1}]]
];

exportarSilvestre[]:=silvester
 
EndPackage[]







(* --- End: libsCP/Quantum2.m --- *)



(* --- Begin: libsCP/Carlos.m --- *)


BeginPackage["Carlos`"] ;


ItppVectorToExpression::usage="Helps read data directly from itpp output, like to interface with a cpp program"

MyAverage::usage = 
    "This function gives the average of more complex quantities than lists, 
    for example it is able to process lists of lists";
RandomUnitVector::usage = "This gives a random vector with the Haar measure. The dimension is 
                           the argument. If no argument is suplied, it is asumed to be 3";

DistanceBetweenSetsOfPoints::usage = "It calculates the distance between two sets of points. They might have a different order. Basically it has been used to compare two spectra";

Seed::usage = 
    "This function, without any argument, gives a random integer between 0
    and 1000000000-1 which can be used as a seed for an external program";

Instruction::usage = 
    "This instrucion creats a string that in s Linux or Unix shell will enter
    some given parameters into an external program. The first argument is a list
    of lists with the values to be entered and the second one is the program";

Norma::usage = "This function gives the norm of a list of numbers";

ColorCoding::usage="This recieves 2 integer inputs, and outputs a 
	graphic to read the number that represents each Hue";
(*
Se comenta porque ya Mathematica 13.1 trae una funcion que hace eso
BlockDiagonalMatrix::usage="Gives a block diagonal matrix from a list"
*)

OffSpellErrors::usage ="Turns of spelling errors"
OnSpellErrors::usage ="Turns on spelling errors"

Log10::usage ="Calculates Log10[x_]:=Log[10,x]"
ColumnAddKeepFirst::usage="To add to matrices, keeping the first column of the first matrix untouched"
ReadListUncomment::usage="igual a ReadList[] pero quita todo lo que comienze con #"
NumberList::usage="Number a list, i.e. prepend with an intenger from 1 to the Length of the list"


HistogramListPoints::usage="Shows the points that would correspond to a Histogram. Accepts
the same options as Histogram and HistogramList. Usage HistogramListPoints[data] or
HistogramListPoints[data, bspec] or HistogramList[data,bspec,hspec]"

HistogramPointsForLine::usage="Calculates the points to make a line corresponding to a Histogram. 
Accepts
the same options as Histogram and HistogramList. Usage HistogramPointsForLine [data] or
HistogramPointsForLine[data, bspec] or HistogramPointsForLine[data,bspec,hspec]"



(* {{{ Symbols and legends *)
MySymbol::usage="Para poner simbolos. Tiene defauls. Es el recomendado ahora"
SymbolNumber::usage="Option for MySymbol"
Coordinate::usage="Option for MySymbol"
Color::usage="Option for MySymbol"
Proportion::usage="Option for MySymbol"
delta::usage="Option for MySymbol"
ThicknessBorder::usage="Option for MySymbol"
MyTriangle::usage = "Graphics almost primitive MyTriangle[{x_, y_}, Color1_, Proportion_, delta_, th_] ";
MyInvertedTriangle::usage = "Graphics almost primitive MyInvertedTriangle[{x_, y_}, Color1_, Proportion_, delta_, th_] ";
MySquare::usage = "Graphics almost primitive MySquare[{x_, y_}, Color1_, Proportion_, delta_, th_]";
MyCircle::usage = "Graphics almost primitive MyCircle[{x_, y_}, Color1_, Proportion_, delta_, th_]";
MyRhombous::usage = "Graphics almost primitive MyRhombous[{x_, y_}, Color1_, Proportion_, delta_, th_]";
My4PointStar1::usage = "Graphics almost primitive My4PointStar1[{x_, y_}, Color1_, Proportion_, delta_, th_]";
My4PointStar2::usage = "Graphics almost primitive My4PointStar2[{x_, y_}, Color1_, Proportion_, delta_, th_]";
My4PointStar3::usage = "Graphics almost primitive My4PointStar3[{x_, y_}, Color1_, Proportion_, delta_, th_]";
My5PointStar::usage = "Graphics almost primitive My5PointStar[{x_, y_}, Color1_, Proportion_, delta_, th_]";

InsetWithSymbols::usage="To create nice symbols in plots. "
MyLegend::usage="Ver LegendBox"

LegendBox::usage=" Ejemplos de uso:
{UpperHeight = 0.9, Xpos = .57, Xlength = .1, 
  XSepText = .1, \[CapitalDelta]Height = .15};
kk = LegendBox[{\"b1\", \"b2\", 
    \"b4\"}, {GrayLevel[0], Style\[Beta][#]} & /@ \[Beta]s, UpperHeight,
    Xpos, Xlength, XSepText, \[CapitalDelta]Height];

LegendBox[
 \"n=\" <> ToString[#] & /@ ns, {Thickness[tjh], Hue[#/Length[ns]]} & /@
   Range[Length[ns]], .9, .6, .2, .05, .1]
"

Alignment::usage="Option for LegendBox and MyLegend"
(* }}} *)
(* {{{ Geometry *)
EllipseCharacteristics::usage="Get center, angle of rotation and semiaxis of an elipse. EllipseCharacteristics[poly_, vars_]
 For example, EllipseCharacteristics[4 x^2 - 4 x y + 7 y^2 + 12 x + 6 y - 9, {x, y}]"



(* }}} *)
Begin["Private`"];


(* Read data from itpp output *)
ItppVectorToExpression[vector_String] := 
 ToExpression /@ StringSplit[ StringReplace[
    StringTake[vector, {2, -2}], {"i" -> "I", "e+" :> "*^", "e-" :> "*^-"}]]
(*  Geometry *)
EllipseCharacteristics[poly_, vars_] :=  (* {{{ *)
 Module[{cl, center, Aq, Bq, Cq, Dq, Eq, Fq, cl2, Am},
  cl = CoefficientList[poly, vars];
  {Aq = cl[[3, 1]], Bq = cl[[2, 2]]/2, Cq = cl[[1, 3]], 
   Dq = cl[[2, 1]]/2, Eq = cl[[1, 2]]/2, Fq = cl[[1, 1]]}; 
  center = -Inverse[{{Aq, Bq}, {Bq, Cq}}].{Dq, Eq};
  cl2 = CoefficientList[poly /. {vars[[1]] -> vars[[1]] + center[[1]], vars[[2]] -> vars[[2]] + center[[2]]}, vars];
  Am = {{cl2[[3, 1]], cl2[[2, 2]]/2}, {cl2[[2, 2]]/2, cl2[[1, 3]]}};
  {center, ArcTan @@ (Eigenvectors[Am][[1]]), 
   1/Sqrt[-Eigenvalues[Am]/cl2[[1, 1]]]}
  ]/; PolynomialQ[poly, vars] (* }}} *)
(*  *)
DistanceBetweenSetsOfPoints[p1_List, p2_List] /; 
  If[Length[p1] == Length[p2], True, 
   Message[DistanceBetweenSetsOfPoints::nnarg, Length[p1], 
    Length[p2]]; False] := 
 Module[{p2tmp = p2, n = Length[p2], OrderedList = {}, k},
  Do[
   k = Nearest[p2tmp[[;; n + 1 - i]] -> Automatic, p1[[i]]];
   OrderedList = Append[OrderedList, p2tmp[[k]][[1]]];
   p2tmp = Drop[p2tmp, k];, {i, n}];
  Total[EuclideanDistance @@@ Transpose[{p1, OrderedList}]]]
DistanceBetweenSetsOfPoints::nnarg = 
  "The lengths of the lists must be equal, but they are  `1` and \
`2`.";

HistogramListPoints[data_, Options___] :=Transpose[{Drop[(#[[1]] + RotateLeft[#[[1]]])/
      2, -1], #[[2]]} &[HistogramList[data, Options]]]


HistogramPointsForLine[data_, Options___] := 
 Module[{hrs = HistogramList[data, Options]},
  Transpose[{Flatten[Transpose[{hrs[[1]], hrs[[1]]}]], 
    Flatten[{0, Transpose[{hrs[[2]], hrs[[2]]}], 0}]}]]


RandomUnitVector[n_] := Module[{v},
  v = RandomReal[NormalDistribution[0, 1], n];
  v/Norm[v]]
RandomUnitVector[] := RandomUnitVector[3]


(*
Se comenta porque ya Mathematica trae una funcion que hace eso
(* From http://mathworld.wolfram.com/BlockDiagonalMatrix.html*)
BlockDiagonalMatrix[b : {__?MatrixQ}] := 
 Module[{r, c, n = Length[b], i, j}, {r, c} = 
   Transpose[Dimensions /@ b];
  ArrayFlatten[
   Table[If[i == j, b[[i]], ConstantArray[0, {r[[i]], c[[j]]}]], {i, 
     n}, {j, n}]]]
*)

NumberList[lista_]:=Flatten[Evaluate[#], 1] & /@ Transpose[{Range[Length[lista]], lista}]

OffSpellErrors[]:={Off[General::spell],Off[General::spell1]}
OnSpellErrors[]:={On[General::spell],On[General::spell1]}

ReadListUncomment[file_, Options___] := 
 ReadList[ StringToStream[
   StringJoin[ StringInsert[#, "\n", -1] & /@  Select[ReadList[file, String], StringFreeQ[#, "#"] &]]], Options]


ColumnAddKeepFirst[MultiList_] :=  MapThread[Prepend, {(Plus @@ MultiList)[[All, 2 ;;]], MultiList[[1, All, 1]]}]
ColumnAddKeepFirst[FirstList_, SecondList_] := ColumnAddKeepFirst[{FirstList, SecondList}]

(*  Legends and symbols {{{ *)
InsetWithSymbols[LowerLeft_List,BoxSize_List,RealtiveCoordinateLowerSymbol_, SepSymbols_,SymbolList_,TextList_, TextSpacing_]:=
	Module[{i},
	{Table[ SymbolList[[i]][ LowerLeft+RealtiveCoordinateLowerSymbol+{0,(i-1) SepSymbols}],{i, Length[SymbolList]}],
      Graphics[ Table[Text[TextList[[i]], LowerLeft+ RealtiveCoordinateLowerSymbol+{0,(i-1) SepSymbols}+{TextSpacing, 0},{-1,0}]
                               ,{i,Length[TextList]}]],
      Graphics[{Line[{LowerLeft,LowerLeft+{0,BoxSize[[2]]},LowerLeft+BoxSize, LowerLeft+{BoxSize[[1]],0},LowerLeft}]}]}
]


MyTriangle[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
    Graphics[{Color1,Polygon[{Scaled[delta {-1/2, -Proportion/3}, {x, y}], 
       Scaled[delta {0, 2 Proportion/3}, {x, y}],Scaled[delta {1/2, -Proportion/3}, {x, y}]}], Thickness[th], 
       GrayLevel[0], Line[{Scaled[delta {-1/2, -Proportion/3}, {x, y}], 
       Scaled[delta {0, 2 Proportion/3}, {x, y}], Scaled[delta {1/2, -Proportion/3}, {x, y}], 
       Scaled[delta {-1/2, -Proportion/3}, {x, y}]}]}];

MyInvertedTriangle[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
    Graphics[{Color1,Polygon[{Scaled[delta {-1/2, Proportion/3}, {x, y}], 
       Scaled[delta {0, -2 Proportion/3}, {x, y}],Scaled[delta {1/2, Proportion/3}, {x, y}]}], Thickness[th], 
       GrayLevel[0], Line[{Scaled[delta {-1/2, Proportion/3}, {x, y}], 
       Scaled[delta {0, -2 Proportion/3}, {x, y}], Scaled[delta {1/2, Proportion/3}, {x, y}], 
       Scaled[delta {-1/2, Proportion/3}, {x, y}]}]}];

MySquare[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
    Graphics[{Color1,  Rectangle[Scaled[delta{-1/2, -Proportion/2}, {x, y}], 
      Scaled[delta{1/2, Proportion/2}, {x, y}]], Thickness[th],GrayLevel[0], 
      Line[{Scaled[delta{-1/2, -Proportion/2}, {x, y}],Scaled[delta{-1/2, Proportion/2}, {x, y}], 
      Scaled[delta{1/2, Proportion/2}, {x, y}], Scaled[delta{1/2, -Proportion/2}, {x, y}], 
      Scaled[delta{-1/2, -Proportion/2}, {x, y}]}]}];

MyRhombous[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
    Graphics[{Color1, 
        Polygon[{Scaled[delta{0, -Proportion/2}, {x, y}], 
            Scaled[delta{1/2, 0}, {x, y}], 
            Scaled[delta{0, Proportion/2}, {x, y}], 
            Scaled[delta{-1/2, 0}, {x, y}]}], Thickness[th], GrayLevel[0], 
        Line[{Scaled[delta{0, -Proportion/2}, {x, y}], 
            Scaled[delta{1/2, 0}, {x, y}], 
            Scaled[delta{0, Proportion/2}, {x, y}], 
            Scaled[delta{-1/2, 0}, {x, y}],  
            Scaled[delta{0, -Proportion/2}, {x, y}]}]}];

My4PointStar1[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
  Module[{PointSet, PointSetLine, theta, alpha},
    alpha = .2;
    PointSet = {{1, 0}, alpha{1, 1}, {0, 1}, alpha{-1, 1}, {-1, 0}, 
        alpha{-1, -1}, {0, -1}, alpha{1, -1}};
    PointSetLine = Flatten[{PointSet, {PointSet[[1]]}}, 1];
    Graphics[{Color1, 
        Polygon[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ 
            PointSet], Thickness[th], GrayLevel[0], 
        Line[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ 
            PointSetLine]}]]

My4PointStar2[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
  Module[{PointSet, PointSetLine, theta, alpha},
    alpha = .3;
    PointSet = {alpha{1, 0}, {1, 1}, alpha{0, 1}, {-1, 1}, 
        alpha{-1, 0}, {-1, -1}, alpha{0, -1}, {1, -1}};
    PointSetLine = Flatten[{PointSet, {PointSet[[1]]}}, 1];
    Graphics[{Color1, 
        Polygon[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ 
            PointSet], Thickness[th], GrayLevel[0], 
        Line[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ 
            PointSetLine]}]]


My5PointStar[{x_, y_}, Color1_, Proportion_, delta_, th_] := Module[{PointSet, PointSetLine, theta,theta2}, 
      PointSet = Flatten[Table[
      {{Cos[theta + Pi/2], Sin[theta + Pi/2]}, 
              1/2 (3 - Sqrt[5]) {Cos[theta + Pi/5 + Pi/2], 
                  Sin[theta + Pi/5 + Pi/2]}}, {theta, 0, 2  Pi - 2  Pi/5, 2  Pi/5}], 1];
     PointSetLine =  Flatten[{PointSet, {PointSet[[1]]}}, 1];
     Graphics[{Color1, Polygon[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ PointSet], 
        Thickness[th], GrayLevel[0], 
          Line[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] &/@ PointSetLine]}]]

MyInverted5PointStar[{x_, y_}, Color1_, Proportion_, delta_, th_] := Module[{PointSet, PointSetLine, theta,theta2}, 
      PointSet = Flatten[Table[
      theta=theta2+Pi;
      {{Cos[theta + Pi/2], Sin[theta + Pi/2]}, 
              1/2 (3 - Sqrt[5]) {Cos[theta + Pi/5 + Pi/2], 
                  Sin[theta + Pi/5 + Pi/2]}}, {theta2, 0, 2  Pi - 2  Pi/5, 2  Pi/5}], 1];
     PointSetLine =  Flatten[{PointSet, {PointSet[[1]]}}, 1];
     Graphics[{Color1, Polygon[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ PointSet], 
        Thickness[th], GrayLevel[0], 
          Line[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] &/@ PointSetLine]}]]

MyCircle[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
  Graphics[{Color1, Disk[{x, y}, Scaled[delta{1/2, Proportion/2}]], 
      Thickness[th], GrayLevel[0], Circle[{x, y}, Scaled[delta{1/2, Proportion/2}]]}]

MyEllipse1[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
  Graphics[{Color1, Disk[{x, y}, Scaled[delta{.5 1/2, Proportion/2 8/5}]], 
      Thickness[th], GrayLevel[0], Circle[{x, y}, Scaled[delta{.5 1/2, Proportion/2 8/5}]]}]
MyEllipse2[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
  Graphics[{Color1, Disk[{x, y}, Scaled[delta{.8, .5 Proportion/2}]], 
      Thickness[th], GrayLevel[0], Circle[{x, y}, Scaled[delta{.8, .5 Proportion/2}]]}]

My4PointStar3[{x_, y_}, Color1_, Proportion_, delta_, th_] := 
 Module[{PointSet, PointSetLine, theta, alpha}, alpha = .3;
  PointSet = {{0, 0}, {Cos[\[Pi]/4 - alpha], 
     Sin[\[Pi]/4 - alpha]}, {Cos[\[Pi]/4 + alpha], 
     Sin[\[Pi]/4 + alpha]}, {0, 0}, {Cos[3 \[Pi]/4 - alpha], 
     Sin[3 \[Pi]/4 - alpha]}, {Cos[3 \[Pi]/4 + alpha], 
     Sin[3 \[Pi]/4 + alpha]}, {0, 0}, {Cos[5 \[Pi]/4 - alpha], 
     Sin[5 \[Pi]/4 - alpha]}, {Cos[5 \[Pi]/4 + alpha], 
     Sin[5 \[Pi]/4 + alpha]}, {0, 0}, {Cos[7 \[Pi]/4 - alpha], 
     Sin[7 \[Pi]/4 - alpha]}, {Cos[7 \[Pi]/4 + alpha], 
     Sin[7 \[Pi]/4 + alpha]}};
  PointSetLine = Flatten[{PointSet, {PointSet[[1]]}}, 1];
  Graphics[{Color1, 
    Polygon[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ 
      PointSet], Thickness[th], GrayLevel[0], 
    Line[Scaled[delta {#[[1]], Proportion #[[2]]}, {x, y}] & /@ 
      PointSetLine]}]]
MyLegend[TheStyle_List, Heigth_, Xpos_, Xlength_, TheText_, XSepText_,  OptionsPattern[]] := 
  {Text[TheText, Scaled[{Xpos + Xlength + XSepText, Heigth}],OptionValue[Alignment]], 
  Join[TheStyle, {Line[{Scaled[{Xpos, Heigth}], 
      Scaled[{Xpos + Xlength, Heigth}]}]}]}

LegendBox[TheLegends_, TheStyles_, UpperHeight_, Xpos_, Xlength_, 
  XSepText_, \[CapitalDelta]Height_,  OptionsPattern[]] := 
 Module[{i}, 
  Table[MyLegend[TheStyles[[i]], 
    UpperHeight - (i - 1) \[CapitalDelta]Height, Xpos, Xlength, 
    TheLegends[[i]], XSepText, Alignment -> OptionValue[Alignment]], {i, Length[TheLegends]}]]
Options[LegendBox] = {Alignment->{0,0}};
Options[MyLegend] = {Alignment->{0,0}};


MySymbol[Coordinate_,  OptionsPattern[]] :=
 {MyTriangle,MySquare,MyRhombous,MyInvertedTriangle,MyCircle,My5PointStar,
 	My4PointStar1,My4PointStar2,MyInverted5PointStar, My4PointStar3, MyEllipse1,MyEllipse2}[[OptionValue[SymbolNumber]]][Coordinate, 
  OptionValue[Color], OptionValue[Proportion], OptionValue[delta], OptionValue[ThicknessBorder]]
Options[MySymbol] = {SymbolNumber -> 1, Color -> Hue[0],
	Proportion -> GoldenRatio, delta -> 0.02,  ThicknessBorder -> 0.001};
(*  }}}  *)

MyAverage[x_] := Plus @@ x/Length[x]

Seed[] := Floor[Random[] 1000000000]

Instruction[jodas_List, Executable_String] := Module[{tmpins},
      tmpins = "printf \""; 
      Do[Do[tmpins = tmpins <> ToString[jodas[[i, j]]] <> " ";, {j, 
            Length[ jodas[[i]] ]}]; 
        tmpins = tmpins <> "\\n";, {i, Length[jodas]}];
      tmpins <> "\" | " <> Executable];

Norma[x_List] := Sqrt[Plus @@( (Abs[x])^2)]


ColorCoding[NumberOfNumbers_Integer,NumberOfColors_Integer]:=
  Module[{n1,n2},n1=2 Pi/NumberOfNumbers;
    n2=2 Pi/NumberOfColors;
    Show[{Graphics[({Hue[#1/(2 Pi-n1)],
                  Text[ToString[N[#1/(2. Pi),2]],1.2 {Cos[#1],Sin[#1]}]}&)/@
            Range[0,2 Pi-n1,n1]],
        Graphics[({Hue[#1/(2 Pi-n2)],Disk[{0,0},1,{#1,#1+n2}]}&)/@
            Range[0,2 Pi-n2,n2]]},DisplayFunction\[Rule]Identity,
      AspectRatio\[Rule]Automatic]]

    
End[];
EndPackage[];




(* --- End: libsCP/Carlos.m --- *)



(* --- Begin: libsCP/CustomTicks.m --- *)


(* :Title: CustomTicks *)
(* :Context: CustomTicks` *)
(* :Author: Mark A. Caprio, Center for Theoretical Physics, 
  Yale University *)
(* :Summary: Custom tick mark generation for linear, log, 
  and general nonlinear axes. *)
(* :Copyright: Copyright 2005, Mark A. Caprio *)
(* :Package Version: 1.2 *)
(* :Mathematica Version: 4.0 *)
(* :History:
      MCAxes package, January 10, 2003.
     MCAxes and then MCTicks packages distributed as part of LevelScheme, 
  2004.
     V1.0, March 11, 2005.  MathSource No. 5599.
     V1.1, March 18, 2005.  Documentation update.
      V1.2, September 17, 2005.  Simplified LogTicks syntax.
  *)



BeginPackage["CustomTicks`"];

Unprotect[Evaluate[$Context<>"*"]];





LinTicks::usage="LinTicks[x1,x2,spacing,subdivisions] or LinTicks[x1,x2] or LinTicks[majorlist,minorlist].";\

LogTicks::usage="LogTicks[power1,power2,subdivisions] or LogTicks[base,power1,power2,subdivisions].";\


TickPreTransformation::usage="Option for LinTicks.  Mapping from given coordinate value to value used for range tests labeling.";\

TickPostTransformation::usage="Option for LinTicks.  Mapping from tick values to actual coordinates used for positioning tick mark, applied after all range tests and labeling.";\

TickRange::usage="Option for LinTicks.";

ShowTickLabels::usage="Option for LinTicks.";
TickLabelRange::usage="Option for LinTicks.";
ShowFirst::usage="Option for LinTicks.";
ShowLast::usage="Option for LinTicks.";
ShowMinorTicks::usage="Option for LinTicks.";
TickLabelStart::usage="Option for LinTicks.";
TickLabelStep::usage="Option for LinTicks.";
TickLabelFunction::usage="Option for LinTicks.";
DecimalDigits::usage="Option for LinTicks.";

MajorTickLength::usage="Option for LinTicks.";
MinorTickLength::usage="Option for LinTicks.";
MajorTickStyle::usage="Option for LinTicks.";
MinorTickStyle::usage="Option for LinTicks.";
MinorTickIndexRange::usage="Option for LinTicks.";
MinorTickIndexTransformation::usage="Option for LinTicks.";

FractionDigits::usage="FractionDigits[x] returns the number of digits to the right of the point in the decimal representation of x.  It will return large values, determined by Precision, for some numbers, e.g., non-terminating rationals.";\

FractionDigitsBase::usage="Option for FractionDigits.";

LimitTickRange::usage="LimitTickRange[{x1,x2},ticks] selects those ticks with coordinates approximately in the range x1...x2.  Ticks must be specified in full form, or at least as a list rather than a bare number.";\

TransformTicks::usage=
    "StripTickLabels[positionfcn,lengthfcn,ticks] transforms the positions and lengths of all tick marks in a list.  Tick marks must be specified in full form, or at least with an explicit pair of in and out lengths.";\

StripTickLabels::usage="StripTickLabels[ticks] removes any text labels from ticks.  Ticks must be specified in full form, or at least as a list rather than a bare number.";\



AugmentTicks::usage="AugmentTicks[labelfunction,lengthlist,stylelist,ticks] augments any ticks in ticklist to full form.";\

AugmentAxisTickOptions::usage="AugmentAxisTickOptions[numaxes,list] replaces any None entries with null lists and appends additional null lists as needed to make numaxes entries.  AugmentAxisTickOptions[numaxes,None] returns a list of null lists.  Note that this differs from the behavior of the Mathematica plotting functions with FrameTicks, for which an unspecified upper or right axis duplicates the given lower or left axis.";\

TickQ::usage="TickQ[x] returns True if x is a valid tick specification." ;
TickListQ::usage="TickListQ[x] returns True if x is a list of valid tick specifications.  This is not a general test for a valid axis tick option specification, as None, Automatic, and a function can also be valid axis tick option specifications.";\
 



LogTicks::oldsyntax="The number of minor subdivisions no longer needs to be specified for LogTicks (see CustomTicks manual for new syntax).";
LogTicks::minorsubdivs="Number of minor subdivisions `1` specified for LogTicks is not 1 or \[LeftCeiling]base\[RightCeiling]-1 (i.e., \[LeftCeiling]base\[RightCeiling]-2 tick marks) and so is being ignored.";\

AugmentTicks::automatic="Tick list must be specified explicitly.";
AugmentAxisTickOptions::numaxes="Tick lists specified for more than `1` axes.";



Begin["`Private`"];



(* range testing and manipulation utilities, from package MCGraphics *)

InRange[{x1_,x2_},x_]:=(x1\[LessEqual]x)&&(x\[LessEqual]x2);

InRange[{x1_,x2_},x_]:=(x1\[LessEqual]x)&&(x\[LessEqual]x2);
InRangeProper[{x1_,x2_},x_]:=(x1<x)&&(x<x2);

InRegion[{{x1_,x2_},{y1_,y2_}},{x_,y_}]:=
    InRange[{x1,x2},x]&&InRange[{y1,y2},y];
InRegion[{x1_,y1_},{x2_,y2_},{x_,y_}]:=InRange[{x1,x2},x]&&InRange[{y1,y2},y];

ExtendRange[PRange:{x1_,x2_},PFrac:{fx1_,fx2_}]:=
    PRange+PFrac*{-1,+1}*-Subtract@@PRange;
ExtendRegion[PRange:{{x1_,x2_},{y1_,y2_}},PFrac:{{fx1_,fx2_},{fy1_,fy2_}}]:=
    PRange+PFrac*{{-1,+1},{-1,+1}}*-Subtract@@@PRange;

(* approximate equality testing utility, from package MCArithmetic *)

Options[ApproxEqual]={Chop\[Rule]1*^-10};
ApproxEqual[x_?NumericQ,y_?NumericQ,Opts___?OptionQ]:=Module[
      {FullOpts=Flatten[{Opts,Options[ApproxEqual]}]},
      Chop[x-y,Chop/.FullOpts]\[Equal]0
      ];
ApproxEqual[x_?NumericQ,_DirectedInfinity]:=False;
ApproxEqual[_DirectedInfinity,x_?NumericQ]:=False;
ApproxInRange[{x1_,x2_},x_,Opts___?OptionQ]:=
    ApproxEqual[x,x1,Opts]||((x1\[LessEqual]x)&&(x\[LessEqual]x2))||
      ApproxEqual[x,x2,Opts];
ApproxIntegerQ[x_?NumericQ,Opts___?OptionQ]:=ApproxEqual[Round[x],x,Opts];







Options[FractionDigits]={FractionDigitsBase\[Rule]10,Limit\[Rule]Infinity,
      Chop\[Rule]1*^-8};

FractionDigits[x_?NumericQ,Opts___?OptionQ]:=Module[
      {
        FullOpts=Flatten[{Opts,Options[FractionDigits]}],
        Value,NumToRight,OptFractionDigitsBase,OptLimit
        },
      Value=N[x];
      OptFractionDigitsBase=FractionDigitsBase/.FullOpts;
      OptLimit=Limit/.FullOpts;
      NumToRight=0;
      While[
        !ApproxIntegerQ[Value,Chop\[Rule](Chop/.FullOpts)]&&(NumToRight<
              OptLimit),
        Value*=OptFractionDigitsBase;
        NumToRight++
        ];
      NumToRight
      ];





NumberSignsOption[x_]:=Switch[
      Sign[x],
      0|(+1),NumberSigns\[Rule]{"",""},
      -1,NumberSigns\[Rule]{"-",""}
      ];

FixedTickFunction[ValueList_List,DecimalDigits_]:=Module[
      {PaddingDecimals,
        PaddingDigitsLeft,PaddingDigitsRight
        },
      PaddingDigitsLeft=If[
          ValueList==={},
          0,
          Length[IntegerDigits[IntegerPart[Max[Abs[ValueList]]]]]
          ];
      PaddingDigitsRight=If[
          DecimalDigits===Automatic,
          If[
            ValueList==={},
            0,
            Max[FractionDigits/@ValueList]
            ],
          DecimalDigits
          ];
      If[PaddingDigitsRight\[Equal]0,
        PaddedForm[IntegerPart[#],PaddingDigitsLeft,NumberSignsOption[#]]&,
        PaddedForm[#,{PaddingDigitsLeft+PaddingDigitsRight,
              PaddingDigitsRight},NumberSignsOption[#]]&
        ]
      ];





Options[LinTicks]={TickPreTransformation\[Rule]Identity,
      TickPostTransformation\[Rule]Identity,ShowFirst\[Rule]True,
      ShowLast\[Rule]True,ShowTickLabels\[Rule]True,ShowMinorTicks\[Rule]True,
      TickLabelStart\[Rule]0,TickLabelStep\[Rule]1,
      TickRange\[Rule]{-Infinity,Infinity},
      TickLabelRange\[Rule]{-Infinity,Infinity},
      TickLabelFunction\[Rule]Automatic,DecimalDigits\[Rule]Automatic,
      MajorTickLength\[Rule]{0.010,0},MinorTickLength\[Rule]{0.005,0},
      MajorTickStyle\[Rule]{},MinorTickStyle\[Rule]{},
      MinorTickIndexRange\[Rule]{1,Infinity},
      MinorTickIndexTransformation\[Rule]Identity};

LinTicks[RawMajorCoordList_List,RawMinorCoordList_List,Opts___]:=Module[
      {
        FullOpts= Flatten[{Opts,Options[LinTicks]}],
        MajorCoordList,
        LabeledCoordList,
        MinorCoordList,
        MajorTickList,
        MinorTickList,
        UsedTickLabelFunction,
        DefaultTickLabelFunction,
        TickValue,
        TickPosition,
        TickLabel,
        TickLength,
        TickStyle,
        i
        },
      
      (* make major ticks *)
      MajorCoordList=
        Select[(TickPreTransformation/.FullOpts)/@RawMajorCoordList,
          ApproxInRange[(TickRange/.FullOpts),#]&];
      LabeledCoordList=Flatten[Table[
            TickValue=MajorCoordList[[i]];
            If[
              (ShowTickLabels/.FullOpts)
                &&ApproxInRange[(TickLabelRange/.FullOpts),TickValue]
                &&(Mod[
                      i-1,(TickLabelStep/.FullOpts)]\[Equal](TickLabelStart/.\
FullOpts))
                &&((i\[NotEqual]1)||(ShowFirst/.FullOpts))
                &&((i\[NotEqual]
                        Length[MajorCoordList])||(ShowLast/.FullOpts)),
              TickValue,
              {}
              ],
            {i,1,Length[MajorCoordList]}
            ]];
      DefaultTickLabelFunction=
        FixedTickFunction[LabeledCoordList,DecimalDigits/.FullOpts];
      UsedTickLabelFunction=Switch[
          (TickLabelFunction/.FullOpts),
          Automatic,(#2&),
          _,(TickLabelFunction/.FullOpts)
          ];
      TickLength=(MajorTickLength/.FullOpts);
      TickStyle=(MajorTickStyle/.FullOpts);
      MajorTickList=Table[
          
          (* calculate tick value *)
          TickValue=MajorCoordList[[i]];
          
          (* calculate coordinate for drawing tick *)
          TickPosition=(TickPostTransformation/.FullOpts)[TickValue];
          
          (* construct label, 
            or null string if it should be suppressed -- if: major tick, 
            
            in designated modular cycle if only a cycle of major ticks are to \
be labeled, tick is in TickLabelRange, 
            and not explicitly suppressed as first or last label; 
            will only then be used if tick is also in TickRange *)
          TickLabel=If[
              (ShowTickLabels/.FullOpts)
                &&ApproxInRange[(TickLabelRange/.FullOpts),TickValue]
                &&(Mod[
                      i-1,(TickLabelStep/.FullOpts)]\[Equal](TickLabelStart/.\
FullOpts))
                &&((i\[NotEqual]1)||(ShowFirst/.FullOpts))
                &&((i\[NotEqual]Length[
                          MajorCoordList])||(ShowLast/.FullOpts)),
              
              UsedTickLabelFunction[TickValue,
                DefaultTickLabelFunction[TickValue]],
              ""
              ];
          
          (* make tick *)
          {TickPosition,TickLabel,TickLength,TickStyle},
          
          {i,1,Length[MajorCoordList]}
          ];
      
      (* make minor ticks *)
      MinorCoordList=
        Select[(TickPreTransformation/.FullOpts)/@RawMinorCoordList,
          ApproxInRange[(TickRange/.FullOpts),#]&];
      TickLength=(MinorTickLength/.FullOpts);
      TickStyle=(MinorTickStyle/.FullOpts);
      MinorTickList=If[(ShowMinorTicks/.FullOpts),
          Table[
            
            (* calculate tick value *)
            TickValue=MinorCoordList[[i]];
            
            (* calculate coordinate for drawing tick *)
            TickPosition=(TickPostTransformation/.FullOpts)[TickValue];
            
            (* make tick *)
            {TickPosition,"",TickLength,TickStyle},
            
            {i,1,Length[MinorCoordList]}
            ],
          {}
          ];
      
      (* combine tick lists*)
      Join[MajorTickList,MinorTickList]
      
      ];

LinTicks[x1_?NumericQ,x2_?NumericQ,Opts___?OptionQ]:=Module[
    {UsedRange,DummyGraphics,TickList,MajorCoordList,MinorCoordList,x},
    
    (* extend any round-number range by a tiny amount *)
    (* this seems to make Mathematica 4.1 give a much cleaner, 
      sparser set of ticks *)
    UsedRange=If[
        And@@ApproxIntegerQ/@{x1,x2},
        ExtendRange[{x1,x2},{1*^-5,1*^-5}],
        {x1,x2}
        ]; 
    
    (* extract raw tick coordinates from Mathematica *)
    DummyGraphics=
      Show[Graphics[{}],PlotRange\[Rule]{UsedRange,Automatic},
        DisplayFunction\[Rule]Identity];
    TickList=First[Ticks/.AbsoluteOptions[DummyGraphics,Ticks]];
    MajorCoordList=Cases[TickList,{x_,_Real,___}\[RuleDelayed]x];
    MinorCoordList=Cases[TickList,{x_,"",___}\[RuleDelayed]x];
    
    (* generate formatted tick mark specifications *)
    LinTicks[MajorCoordList,MinorCoordList,Opts]
    ]

LinTicks[x1_?NumericQ,x2_?NumericQ,Spacing_?NumericQ,MinorSubdivs:_Integer,
      Opts___]:=Module[
      {
        FullOpts= Flatten[{Opts,Options[LinTicks]}],
        MaxMajorIndex,
        MajorCoordList,MinorCoordList
        },
      
      (* preliminary calculations  *)
      MaxMajorIndex=Round[(x2-x1)/Spacing];
      
      (* construct table of ticks --indexed by MajorIndex=0,1,...,
        MaxMajorTick and MinorIndex=0,...,MinorSubdivs-1, 
        where MinorIndex=0 gives the major tick, 
        except no minor ticks after last major tick *)
      MajorCoordList=Flatten[Table[
            N[x1+MajorIndex*Spacing+MinorIndex*Spacing/MinorSubdivs],
            {MajorIndex,0,MaxMajorIndex},{MinorIndex,0,0}
            ]];
      MinorCoordList=Flatten[Table[
            If[
              InRange[MinorTickIndexRange/.FullOpts,MinorIndex],
              
              N[x1+MajorIndex*
                    Spacing+((MinorTickIndexTransformation/.FullOpts)@
                        MinorIndex)*Spacing/MinorSubdivs],
              {}
              ],
            {MajorIndex,0,MaxMajorIndex},{MinorIndex,1,MinorSubdivs-1}
            ]];
      
      (* there are usually ticks to be suppressed at the upper end, 
        since the major tick index rounds up to the next major tick (for \
safety in borderline cases where truncation might fail), 
        and the loop minor tick index iterates for a full series of minor \
ticks even after the last major tick *)
      MajorCoordList=Select[MajorCoordList,ApproxInRange[{x1,x2},#]&];
      MinorCoordList=Select[MinorCoordList,ApproxInRange[{x1,x2},#]&];
      
      LinTicks[MajorCoordList,MinorCoordList,Opts]
      
      ];





LogTicks[Base:(_?NumericQ):10,p1_Integer,p2_Integer,Opts___?OptionQ]:=Module[
      {BaseSymbol,MinorSubdivs},
      
      BaseSymbol=If[Base===E,StyleForm["e",FontFamily->"Italic"],Base];
      MinorSubdivs=Ceiling[Base]-1; (* one more than minor ticks *)
      MinorSubdivs=Max[MinorSubdivs,1]; (* 
        prevent underflow from bases less than 2 *)
      LinTicks[p1,p2,1,MinorSubdivs,
        TickLabelFunction\[Rule](DisplayForm[
                SuperscriptBox[BaseSymbol,IntegerPart[#]]]&),
        MinorTickIndexTransformation\[Rule](Log[Base,#+1]*MinorSubdivs&),
        Opts
        ]
      
      ];

(* syntax traps for old syntax -- but will not catch usual situation in which \
base was unspecified but subdivs was *)
LogTicks[Base_?NumericQ,p1_Integer,p2_Integer,MinorSubdivs_Integer,
        Opts___?OptionQ]/;(MinorSubdivs\[Equal]
          Max[Ceiling[Base]-1,1]):=(Message[LogTicks::oldsyntax];
      LogTicks[Base,p1,p2,ShowMinorTicks\[Rule]True,Opts]);
LogTicks[Base_?NumericQ,p1_Integer,p2_Integer,MinorSubdivs_Integer,
        Opts___?OptionQ]/;(MinorSubdivs==1):=(Message[LogTicks::oldsyntax];
      LogTicks[Base,p1,p2,ShowMinorTicks\[Rule]False,Opts]);
LogTicks[Base_?NumericQ,p1_Integer,p2_Integer,MinorSubdivs_Integer,
        Opts___?OptionQ]/;((MinorSubdivs!=
              Max[Ceiling[Base]-1,1])&&(MinorSubdivs!=1)):=(Message[
        LogTicks::oldsyntax];Message[LogTicks::minorsubdivs,MinorSubdivs];
      LogTicks[Base,p1,p2,ShowMinorTicks\[Rule]True,Opts]);





AugmentTick[LabelFunction_,DefaultLength_List,DefaultStyle_List,x_?NumericQ]:=
    AugmentTick[LabelFunction,DefaultLength,DefaultStyle,{x}];



AugmentTick[LabelFunction_,DefaultLength_List,
      DefaultStyle_List,{x_?NumericQ}]:=
    AugmentTick[LabelFunction,DefaultLength,DefaultStyle,{x,LabelFunction@x}];



AugmentTick[LabelFunction_,DefaultLength_List,
      DefaultStyle_List,{x_?NumericQ,LabelText_}]:=
    AugmentTick[LabelFunction,DefaultLength,
      DefaultStyle,{x,LabelText,DefaultLength}];



AugmentTick[LabelFunction_,DefaultLength_List,
      DefaultStyle_List,{x_?NumericQ,LabelText_,PLen_?NumericQ,RestSeq___}]:=
    AugmentTick[LabelFunction,DefaultLength,
      DefaultStyle,{x,LabelText,{PLen,0},RestSeq}];



AugmentTick[LabelFunction_,DefaultLength_List,
      DefaultStyle_List,{x_?NumericQ,LabelText_,LengthList_List}]:=
    AugmentTick[LabelFunction,DefaultLength,
      DefaultStyle,{x,LabelText,LengthList,DefaultStyle}];



AugmentTick[LabelFunction_,DefaultLength_List,DefaultStyle_List,
      TheTick:{x_?NumericQ,LabelText_,LengthList_List,Style_List}]:=TheTick;



AugmentTicks[DefaultLength_List,DefaultStyle_List,TickList_List]:=
    AugmentTick[""&,DefaultLength,DefaultStyle,#]&/@TickList;





AugmentAxisTickOptions[NumAxes_Integer,TickLists:None]:=Table[{},{NumAxes}];
AugmentAxisTickOptions[NumAxes_Integer,TickLists_List]:=Module[
      {},
      If[
        NumAxes<Length[TickLists],
        Message[AugmentAxisTickOptions::numaxes,NumAxes]
        ];
      Join[
        Replace[TickLists,{None\[Rule]{}},{1}],
        Table[{},{NumAxes-Length[TickLists]}]
        ]
      ];





TickPattern=(_?NumericQ)|{_?NumericQ}|{_?NumericQ,_}|{_?
          NumericQ,_,(_?NumericQ)|{_?NumericQ,_?NumericQ}}|{_?
          NumericQ,_,(_?NumericQ)|{_?NumericQ,_?NumericQ},_List};

TickQ[x_]:=MatchQ[x,TickPattern];
TickListQ[x_]:=MatchQ[x,{}|{TickPattern..}];















LimitTickRange[Range:{x1_,x2_},TickList_List]:=
    Cases[TickList,{x_,___}/;ApproxInRange[{x1,x2},x]];

TransformTicks[PosnTransformation_,LengthTransformation_,TickList_List]:=
    Replace[TickList,{x_,t_,l:{_,_},
          RestSeq___}\[RuleDelayed]{PosnTransformation@x,t,
          LengthTransformation/@l,RestSeq},{1}];

StripTickLabels[TickList_List]:=
    Replace[TickList,{x_,_,RestSeq___}\[RuleDelayed]{x,"",RestSeq},{1}];



End[];



Protect[Evaluate[$Context<>"*"]];
EndPackage[];

















































































































































































(* --- End: libsCP/CustomTicks.m --- *)



(* --- Begin: libsCP/Quantum.m --- *)


(* ::Package:: *)

(* {{{ *) BeginPackage["Quantum`"]
(* {{{ TODO
Crear un swap matrix en el mismo espiritu que la CNOT. 
}}} *)
(* {{{ Bitwise manipulation and basic math*)
ExtractDigits::usage = "Extract the digits of a number in its binary form. Same in spirity as exdig in math.f90
   with numin is NumberIn and nwhich is Location digits,
   This routine takes numin and puts two numbers
   n1out and n2out which result from the digits
   of numin that are marked with the 1 bits
   of the number nwhich
   nwhich=   0 1 0 1 0 0 1 = 42
   numin=    0 1 1 0 1 1 1 = 55
   n1out=      1   0     1 = 5
   n2out=    0   1   1 1   = 7, {n1out,n2out}"
BitsOn::usage = "The number of 1s in a binary representation, say 
   n = 5 = 1 0 1
   BitsOn[n] = 2"
MergeTwoIntegers::usage = "MergeTwoIntegers[na_, nb_, ndigits_]
  Merge two intergers based on the positions given by the 1s of the 
  third parameter. Functions similar to the one written in fortran, merge_two_integers.
  In this routine we merge two numbers. It is useful for doing the tensor
  product. 'a' and 'b' are the input number wheares as usual ndigits
  indicates the position. Example
  ndigits            = 0 0 1 0 1 0 0 1 = 41
  a                  =     1   0     1 = 5
  b                  = 1 0   0   1 0   = 18
  merge_two_integers = 1 0 1 0 0 1 0 1 = 165"
xlog2x::usage = "Calculates x Log[2, x], but if x is 0, takes the correct limit"

GetDigitQuantumPosition::usage = " GetDigitQuantumPosition[index_, qubitPosition_] 
Calculates the digit of an index, assuming that
the array starts in 0 (as we count in quantum information) and that the position 
of the qubit is 0 at the rightmost position. Example:
Table[GetDigit[5, i], {i, 0, 5}] -> {1, 0, 1, 0, 0, 0}"


(* }}} *)
(* {{{ Random States and matrices*)
TwoRandomOrhtogonalStates::usage = "TwoRandomOrhtogonalStates[dim_] creates two random states orthogonal to each other using gram schmidt process. It is useful for creating states that provide a maximally mixed state in a qubit"
RandomState::usage = "A random state RandomState[dim_Integer]"
RandomGaussianComplex::usage = "RandomGaussianComplex[] gives a complex random number with Gaussian distribution
   centered at 0 and with width 1"
RandomGaussian::usage = "RandomGaussian[] gives a random number with Gaussian distribution
   centered at 0 and with width 1"
RandomHermitianMatrix::usage = "RandomHermitianMatrix[ ] To generate a GUE Random Hermitian Matrix, various normalizatios are possible"
PUEMember::usage = "PUEMember[n_] To generate a PUE Random Hermitian Matrix, with spectral span from -1/2 to 1/2. n is the dimension"
GUEMember::usage = "GUEMember[n_] To generate a GUE Random Hermitian Matrix, various normalizatios are possible. n is the dimension"
GOEMember::usage = "GOEMember[n_] To generate a GOE Random Hermitian Matrix, various normalizatios are possible. n is the dimension"
CUEMember::usage = "To generate a CUE Random Unitary Matrix: CUEMember[n_]. n is the dimension"
RandomDensityMatrix::usage = "RandomDensityMatrix[n_] to generate a random density matrix of length n"
Normalization::usage = "default sucks, SizeIndependent yields  Hij Hkl = delta il delta jk, kohler gives
                          = N/Pi^2 delta il delta jk" 
(* }}} *)
(* {{{ Matrix manipulation and advanced linear algebra*)
BaseForHermitianMatrices::usage = "Base element for Hermitian matrices"
Commutator::usage = "Makes the commutator between A and B: Commutator[A,B]=A.B-B.A"
PartialTrace::usage = "Yields the partial trace of a multiple qubit state. The second argument represents the qubit states to be left 
in binary. For example, suppose you have five qubits and you want the first and last qubit to be left, you would write:
           1  0  0  0  1
Position:  4  3  2  1  0    
Then, the second argument would be 17. 
PartialTrace[Rho_?MatrixQ, DigitsToLeave_] or PartialTrace[Psi_?VectorQ, LocationDigits_]"
PartialTranspose::usage = "Takes the partial transpososition with  respect
  to the indices specified, PartialTranspose[A_, TransposeIndices_]"
PartialTransposeFirst::usage = "Transpose the first part of a bipartite system with equal dimensions"
PartialTransposeSecond::usage = "Transpose the Second part of a bipartite system with equal dimensions"
DirectSum::usage = "DirectSum[A1, A2] A1 \[CirclePlus] A2 or DirectSum[{A1, A2, ..., An}] = A1 \[CirclePlus] A2\[CirclePlus] ... \[CirclePlus] An"
tensorProduct::usage = "Creates the tensor product for qubits, like TensorProduct_real_routine in linear.f90
      I think that the inidices in the number refer to where to place the second matrix."
tensorPower::usage = "Creates the tensor product of n matrices" (*JA: est\[AAcute] extra\[NTilde]a esta definici\[OAcute]n*)
OrthonormalBasisContaningVector::usage=" OrthonormalBasisContaningVector[psi_?VectorQ] will create an orthonorlam basis that contains the given vector"
GetMatrixForm[Gate_, Qubits_] := Gate /@ IdentityMatrix[Power[2, Qubits]]
ArbitraryPartialTrace::usage="Yields the partial trace of a Matrix. The first entry is the dimension of the Hilbert Space that you want
to remove or trace out. The second is the dimension of the Hilbert space that you want to keep. The third is the Matrix on which this 
operation will be applied."
(* }}} *)
(* {{{ Quantum information routines *)
ApplyControlGate::usage = "ApplyControlGate[U_, State_, TargetQubit_, ControlQubit_]"
ApplyGate::usage = "ApplyGate[U_, State_, TargetQubit_]"
ApplyOperator::usage = "ApplyOperator[op,rho] applies the operator op to the density matrix rho."
ApplyChannel::usage = "Apply Quantum Channel to an operator"
ControlNotMatrix::usage = "Get the control not matrix ControlNotMatrix[QubitControl_Integer, QubitTarget_Integer, NumberOfQubits_Integer]"
SWAPMatrix::usage = "SWAPMatrix[n,{j,k}] yields the matrix that swaps the jth and kth qubits in a system of n qubits"
QuantumDotProduct::usage = "Dot Product in which the first thing is congutate. Yields the true Psi1 dot Psi2" 
ToSuperOperatorSpaceComputationalBasis::usage = "ToSuperOperatorSpaceComputationalBasis[rho_] converts a density matrix rho to superoperator space, where it is a vector, and the basis in the superoperator space is the computational basis, which is the same (or at least similar) to the Choi basis"
FromSuperOperatorSpaceComputationalBasis::usage = "FromSuperOperatorSpaceComputationalBasis[rho_?VectorQ] converts a density matrix rho from superoperator space, where it is a vector, and the basis in the superoperator space is the computational basis, which is the same (or at least similar) to the Choi basis to the normal space, where it is a matrix"
FromSuperOperatorSpacePauliBasis::usage = "FromSuperOperatorSpacePauliBasis[r_?VectorQ] maps a vetor to a density, as the inverse of ToSuperOperatorSpacePauliBasis"
ToSuperOperatorSpacePauliBasis::usage = "ToSuperOperatorSpacePauliBasis[r_?MatrixQ] maps a density
matrix to superoperator space using the Pauli matrices and tensor products"
BlochNormalFormVectors::usage = "Vectors to calculate normal form, see arXiv:1103.3189v1"
BlochNormalFormFromVectors::usage = "Density matrix in normal form, see arXiv:1103.3189v1"
StateToBlochBasisRepresentation::usage = "The so called Bloch matrix, see arXiv:1103.3189v1"
CoherentState::usage = "CoherentState[\[Theta]_, \[Phi]_, n_] Makes a Coherent Spin state where the inputs are the Bloch angles and the number of qubits"
Pauli::usage = "Pauli[0-3] gives Pauli Matrices according to wikipedia, and Pauli[{i1,i2,...,in}] gives Pauli[i1] \[CircleTimes]Pauli[i2] \[CircleTimes] ... \[CircleTimes] Pauli[in]"
Paulib::usage = "Pauli[b_] gives b.Sigma, if b is a 3 entry vector"
SumSigmaX::usage = "Matrix correspoding to sum_j Pauli[1]_j"
SumSigmaY::usage = "Matrix correspoding to sum_j Pauli[2]_j"
SumSigmaZ::usage = "Matrix correspoding to sum_j Pauli[3]_j"
ValidDensityMatrix::usage = "Test whether a given density matrix is valid"
Dagger::usage = "Hermitian Conjugate"
Concurrence::usage = "Concurrence of a 2 qubit density matrix density matrix"
MultipartiteConcurrence::usage = "Concurrence of multipartite entanglement for pure states"
Purity::usage = "Purity of a 2 qubit density matrix density matrix"
Proyector::usage = "Gives the density matrix rho=|Psi><Psi| corresponging to |Psi>, or rho=|Phi><Psi| se se le dan dos argumentos: Proyector[Phi, Psi]"
KrausOperatorsForSuperUnitaryEvolution::usage = "Gives the Kraus Operators for a given unitary and a given state of the environement"
ApplyKrausOperators::usage = "Apply a set of Kraus Operors, (for example fro the output of KrausOperatorsForSuperUnitaryEvolution to a state. Synthaxis is ApplyKrausOperators[Operators_, rho_]. "
vonNeumannEntropy::usage = "von Neumann entropy of a density Matrix, or a list of vectors. Usage vonNeumannEntropy[r_]"
Bell::usage = "Bell[n] state gives a maximally entangled state of the type 
sum_i^n |ii\>. Bell[ ] gives |00> + |11>" 
StateToBlochSphere::usage="Get the Bloch sphere point of a mixed state StateToBlochSphere[R_?MatrixQ]. Gives the components in cartesian coordinates"
BlochSphereToState::usage="BlochSphereToState[CartesianCoordinatesOfPoint_List] transforms the points in the Bloch Sphere to a mixed state. Alternatively, one can give the angles, so BlochSphereToState[{\[Theta]_, \[Phi]_}]"
QuantumMutualInformationReducedEntropies::usage="QuantumMutualInformationReducedEntropies[r_?MatrixQ] calcula la informacion mutia con entropias"
QuantumMutualInformationMeasurements::usage="QuantumMutualInformationMeasurements[r_?MatrixQ] calcula la informacion mutua cuando 
permitimos unla entropia condicional de una matriz de dos qubits cuando hacemos una medicion sobre un qubit. "
Discord::usage="Discord[r_?MatrixQ] calcula el quantum discord de una matriz de dos qubits."
BasisState::usage="BasisState[BasisNumber_,Dimension_] gives a state of the computational basis. It is 
numbered from 0 to Dimension-1"
Hadamard::usage="Hadamard[] gate, Hadamard[QubitToApply, Total"
LocalOneQubitOperator::usage="LocalOneQubitOperator[n,k,A] creates an operator that applies the one-qubit operator A at position k in a n-qubit system."
(* }}} *)
(* {{{ Quantum channels and basis transformations *)
JamiolkowskiStateToOperatorChoi::usage = "JamiolkowskiStateToOperatorChoi[Rho] applies the Jamiolkovski isomorphism as is understood in Geometry of Quantum States pgs. 241, 243, 266"
JamiolkowskiOperatorChoiToState::usage = "JamiolkowskiOperatorChoiToState[O] applies the inverse Jamiolkovski isomorphism as is understood in Geometry of Quantum States"
TransformationMatrixPauliBasisToComputationalBasis::usage = "The matrix that allows to tranform, in superoperator space, from the Pauli basis (the GellMann basis for dimension 2 modulo order) to the computational basis, aka the Choi basis"
Reshuffle::usage = "Apply the reshufle operation as undestood in Geometry of Quantum States, pags 260-262, and 264"
RandomTracePreservingMapChoiBasis::usage = "Creates a singlequbit random trace preserving map"
AveragePurityChannelPauliBasis::usage = "Calculates the average final purity given that the initial states are pure, and chosen with the Haar measure. "
BlochEllipsoid::usage = "BlochEllipsoid[Cha_] Shows the deformation of Bloch sphere for a qubit channel in the pauli basis."
EvolvGate::usage="EvolvGate[Gate_, steps_, env_, state_]... Evoluciona cualquier estado aplicando un numero steps_ de veces la compuerta Gate de  la forma Gate[#, otherparameters_] donde debe ponerse # en el lugar donde Gate toma el estado"
MakeQuantumChannel::usage="MakeQuantumChannel[Gate_, steps_, env_] Donde Gate va de la forma Gate[#, otherparameters_]  donde debe ponerse # en el lugar donde Gate toma el estado"
SumPositiveDerivatives::usage="SumPositiveDerivatives[list_] Suma todas las contribuciones list(max)-list(min) sucesivos cuando la derivada es positiva"
GHZ::usage="GHZ[qu_] Creates a GHZ state, |000000\[RightAngleBracket]+|111111\[RightAngleBracket]"
Wstate::usage="Wstate[n_] Creates a W state, (|10...0>+|010...0>+...+|0...01>)/sqrt{n}"
RandomMixedState::usage="RandomMixedState[n_,combinations_], Constructs a Random mixed state of diemnsion n with random uniform combinations of pure states wti Haar measure."
GellMann::usage = "GellMann[n_] Generalized Gell-Mann matrices from https://blog.suchideas.com/2016/04/sun-gell-mann-matrices-in-mathematica/ For example
for n=2 it gives Pauli matrices, don't forget to add identity by yourself in need a complete basis."
ApplySwap::usage= "ApplySwap[State,{j,k}] applies swap map betwen the jth and kth qubits, the input can be either a state vector or a density matrix."
ApplySwapPure::usage = "Leaves the state in ket form if pure"
ApplyLocalNoiseChain::usage = "ApplyLocalNoiseChain[State,p] Applies the map that transforoms the density matrix State into the assessible density matrix when local noise is present using fuzzy measurements."
ApplyNoiseChain::usage = "ApplyNoiseChain[State,p] Applies the map that transforoms the density matrix State into the assessible density matrix when non-local noise is present using fuzzy measurements."
PermutationMatrices::usage = "Argument is the number of particles to permute, output is a list of matrices."
(*
Commented as is provided by Mathematica in 13.1
PermutationMatrix::usage = "PermutationMatrix[p_List]."
*)
PauliSuperoperator::usage="Calculates superoperator of a Pauli quantum channel. Pauli quantum channels transform 
qubits density matrices as \!\(\*SubscriptBox[\(r\), \(\*SubscriptBox[\(j\), \(1\)],  ... , \*SubscriptBox[\(j\), \(n\)]\)]\)\[Rule]\!\(\*SubscriptBox[\(\[Tau]\), \(\*SubscriptBox[\(j\), \(1\)],  ... , \*SubscriptBox[\(j\), \(n\)]\)]\)\!\(\*SubscriptBox[\(r\), \(\*SubscriptBox[\(j\), \(1\)],  ... , \*SubscriptBox[\(j\), \(n\)]\)]\), where \!\(\*SubscriptBox[\(r\), \(\*SubscriptBox[\(j\), \(1\)],  ... , \*SubscriptBox[\(j\), \(n\)]\)]\) are the components 
of \[Rho] in Pauli tensor products basis.
PauliSuperoperator[pauliDiagonal_List]"
PCEFigures::usage="
Returns the figure representing the Pauli channel erasing operation. Works up 
to 3 qubits (up to cubes). Parameters: list of 1D, 2D or 3D correlations left
invariant by a PCE operation.
PCEFigures[correlations_List]"
ApplyMultiQubitGate::usage="ApplyMultiQubitGate[state_,gate_,targets_] Applies multiqubit gate, targets in binary. State can be a vector or a density matrix."
(* }}} *)
(* }}} *)
Begin["Private`"] 
(* {{{ Bitwise manipulation and basic math*)
(* {{{ *) GetDigitQuantumPosition[index_, qubitPosition_] := 
 Module[{digits = Reverse[IntegerDigits[index, 2]]},
  If[qubitPosition >= Length[digits], 0, digits[[qubitPosition + 1]]]]
(* }}} *)
(* {{{ *) ExtractDigits[NumberIn_, LocationDigits_] := Module[{AuxArray, NumberOfDigits}, 
	NumberOfDigits = IntegerLength[NumberIn, 2]; 
	AuxArray = Transpose[{IntegerDigits[LocationDigits, 2, NumberOfDigits], 
	IntegerDigits[NumberIn, 2, NumberOfDigits]}];
	{FromDigits[Select[AuxArray, #[[1]] != 1 &][[All, 2]], 2], 
	FromDigits[Select[AuxArray, #[[1]] == 1 &][[All, 2]], 2]}] 

(* }}} *)
(* {{{ *) BitsOn[n_] := Count[IntegerDigits[n, 2], 1]
(* }}} *)
(* {{{ *) MergeTwoIntegers[na_, nb_, ndigits_] := 
 Module[{LongitudTotal, Digits01s, Result}, 
  LongitudTotal = 
   Max[Count[IntegerDigits[ndigits, 2], 0], BitLength[nb]] + 
    BitsOn[ndigits];
  Digits01s = 
   PadRight[Reverse[IntegerDigits[ndigits, 2]], LongitudTotal];
  Result = PadRight[{}, LongitudTotal];
  Result[[Flatten[Position[Digits01s, 1]]]] = 
   Reverse[IntegerDigits[na, 2, Count[Digits01s, 1]]];
  Result[[Flatten[Position[Digits01s, 0]]]] = 
   Reverse[IntegerDigits[nb, 2, Count[Digits01s, 0]]];
  FromDigits[Reverse[Result], 2]]


(* }}} *)
(* {{{ *) xlog2x[x_] := If[x == 0, 0, x Log[2, x]]

(* }}} *)
(* }}} *)
(* {{{ Random States and matrices*)
(* {{{ *) TwoRandomOrhtogonalStates[dim_] := Module[{psi1, psi2, prepsi2},
  psi1 = RandomState[dim];
  prepsi2 = RandomState[dim];
  prepsi2 = prepsi2 - Dot[Conjugate[psi1], prepsi2] psi1;
  psi2 = #/Norm[#] &[prepsi2];
  {psi1, psi2}]
(* }}} *)
(* {{{ *) RandomState[dim_Integer] := #/(Sqrt[Conjugate[#] . #])&[Table[ RandomGaussianComplex[], {i, dim}]]; 
(* }}} *)
(* {{{ *) RandomGaussianComplex[] := #[[1]] + #[[2]] I &[RandomReal[NormalDistribution[], {2}]];
(* }}} *)
(* {{{ *) RandomGaussianComplex[A_,sigma_] := sigma(#[[1]] + #[[2]] I &[RandomReal[NormalDistribution[], {2}]])+A;
(* }}} *)
(* {{{ *) RandomGaussian[] :=RandomReal[NormalDistribution[0, 1]];
(* }}} *)
(* {{{ *) RandomHermitianMatrix[qubits_Integer,   OptionsPattern[]] :=
	Switch[OptionValue[Normalization],"Default",1,"SizeIndependent",0.5, "Kohler", 
	Power[2,qubits-1]/Power[Pi,2]] ((Transpose[Conjugate[#]] + #)&) @@ 
	{Table[ RandomGaussianComplex[], {i, Power[2, qubits]}, {j, Power[2, qubits]}]};
Options[RandomHermitianMatrix] = {Normalization -> "Default"};

(* }}} *)
(* {{{ *) PUEMember[n_Integer] := Module[{U},
	U = CUEMember[n];
	Chop[U . DiagonalMatrix[Table[Random[], {n}]-0.5] . Dagger[U]]]
(* }}} *)
(* {{{ *) GUEMember[n_Integer,   OptionsPattern[]] :=
	Times[Switch[OptionValue[Normalization],"Default",1,"SizeIndependent",0.5, "Kohler", (n/2)/Power[Pi,2]], 
	((Transpose[Conjugate[#]] + #)&) @@ {Table[ RandomGaussianComplex[], {i, n}, {j, n}]}];
Options[GUEMember] = {Normalization -> "Default"};

(* }}} *)
(* {{{ *) GOEMember[n_Integer,   OptionsPattern[]] :=
(* With the normalization "f90" we get consistent values as with the module *)
(* In particular, we obtain that <W_{ij} W_{kl}>=\delta_{il}\delta_{jk}+\delta{ik}\delta_{jl} *)
	Times[Switch[OptionValue[Normalization],"Default",1,"f90",1/Sqrt[2],"SizeIndependent",0.5, "Kohler", (n/2)/Power[Pi,2]], 
	((Transpose[#] + #)&) @@ {Table[ RandomGaussian[], {i, n}, {j, n}]}];
Options[GOEMember] = {Normalization -> "Default"};

(* }}} *)
(* {{{ *) CUEMember[n_] := Transpose[ Inner[Times, Table[Exp[I Random[Real, 2 \[Pi]]], {n}], 
	Eigenvectors[GUEMember[n]], List]]


(* }}} *)
(* {{{ *) RandomDensityMatrix[n_] := #/Tr[#] &[(# . Transpose[Conjugate[#]]) &[GUEMember[n]]]

(* }}} *)
(* }}} *)
(* {{{ Matrix manipulation and advanced linear algebra*)
(* {{{ *) BaseForHermitianMatrices[j_Integer, Ncen_Integer] := 
 If[j <= Ncen, SparseArray[{j, j} -> 1, {Ncen, Ncen}], 
  SparseArray[{{Ncen - Ceiling[TriangularRoot[j - Ncen]], 
      1 + Ncen - (j - Ncen - Triangular[Ceiling[TriangularRoot[j - Ncen]] - 1])}, 
      {1 + Ncen - (j - Ncen - Triangular[Ceiling[TriangularRoot[j - Ncen]] - 1]), 
      Ncen - Ceiling[TriangularRoot[j - Ncen]]}} -> 1/Sqrt[2], {Ncen, Ncen}]]
(* {{{ TriangularRoot and Triangular*) 
TriangularRoot[n_] := (-1 + Sqrt[8 n + 1])/2
Triangular[n_] := n (n + 1)/2
(* }}} *)

(* }}} *)
(* {{{ *) Commutator[A_?MatrixQ, B_?MatrixQ] := A . B - B . A
(* }}} *)
(* {{{ *) PartialTrace[Rho_?MatrixQ, DigitsToLeave_] := Module[{ab1, ab2, na, nb},
	nb = Power[2, BitsOn[DigitsToLeave]];
	na = Length[Rho]/nb;
	Table[
		Sum[
		(*Print[a,b1,n,DigitsToLeave,{MergeTwoIntegers[b1-1,a-1,
		 DigitsToLeave],MergeTwoIntegers[a-1,b1-1,n]},{MergeTwoIntegers[
		 b2-1,a-1,DigitsToLeave],MergeTwoIntegers[a-1,b2-1,n]}];*)

		ab1 = MergeTwoIntegers[b1 - 1, a - 1, DigitsToLeave];
	ab2 = MergeTwoIntegers[b2 - 1, a - 1, DigitsToLeave];
	Rho[[ab1 + 1, ab2 + 1]], {a, na}],
		{b1, nb}, {b2, nb}]]
(* }}} *)
(* {{{ *) PartialTrace[Psi_?VectorQ, LocationDigits_] := Module[{DimHCentral, MatrixState, i}, 
	DimHCentral = Power[2, DigitCount[LocationDigits, 2, 1]];
	MatrixState = SparseArray[Table[1 + ExtractDigits[i - 1, LocationDigits] -> Psi[[i]], {i,Length[Psi]}]];
	Table[QuantumDotProduct[MatrixState[[All, j2]], MatrixState[[All, j1]]], 
		{j1, DimHCentral}, {j2, DimHCentral}]]


(* }}} *)
(* {{{ *) PartialTranspose[A_?MatrixQ, TransposeIndices_Integer] := 
 Module[{i, j, k, l, il, kj},
  Table[
   {i, j} = {BitAnd[ij - 1, BitNot[TransposeIndices]], 
     BitAnd[ij - 1, TransposeIndices]};
   {k, l} = {BitAnd[kl - 1, BitNot[TransposeIndices]], 
     BitAnd[kl - 1, TransposeIndices]};
   {il, kj} = {i + l, k + j};
   A[[il + 1, kj + 1]], {ij, Length[A]}, {kl, Length[A]}]]
(* }}} *)
(* {{{ *) PartialTransposeFirst[A_?MatrixQ] := 
 Module[{n, i, j, k, l, ij, kl, kj, il},
  n = Sqrt[Length[A]];
  Table[
   {i, j} = IntegerDigits[ij, n, 2];
   {k, l} = IntegerDigits[kl, n, 2];
   kj = FromDigits[{k, j}, n];
   il = FromDigits[{i, l}, n];
   A[[kj + 1, il + 1]], {ij, 0, n^2 - 1}, {kl, 0, n^2 - 1}]
  ]
(* }}} *)
(* {{{ *) PartialTransposeSecond[A_?MatrixQ] := 
 Module[{n, i, j, k, l, ij, kl, kj, il},
  n = Sqrt[Length[A]];
  Table[
   {i, j} = IntegerDigits[ij, n, 2];
   {k, l} = IntegerDigits[kl, n, 2];
   kj = FromDigits[{k, j}, n];
   il = FromDigits[{i, l}, n];
   A[[il + 1, kj + 1]], {ij, 0, n^2 - 1}, {kl, 0, n^2 - 1}]
  ]
(* }}} *)
(* {{{ *) DirectSum[MatrixList_List] := Fold[DirectSum, MatrixList[[1]], Drop[MatrixList, 1]]
(* }}} *)
(* {{{ *) DirectSum[A1_, A2_] := Module[{dims},
  dims = Dimensions /@ {A1, A2};
  ArrayFlatten[{{A1, Table[0, {dims[[1, 1]]}, {dims[[2, 2]]}]},
    {Table[0, {dims[[2, 1]]}, {dims[[1, 2]]}], A2}}]]
(* }}} *)
(* {{{ *) tensorProduct[Matrix1_?MatrixQ, Matrix2_?MatrixQ] := KroneckerProduct[Matrix1, Matrix2]
(* }}} *)
(* {{{ *) tensorProduct[LocationDigits_Integer, Matrix1_?MatrixQ, Matrix2_?MatrixQ] := 
	Module[{Indices, iRow, iCol, L1, L2}, 
		{L1,L2}=Length/@{Matrix1,Matrix2};
		Normal[SparseArray[Flatten[Table[
			Indices = {1 + ExtractDigits[iRow - 1, LocationDigits], 
			1 + ExtractDigits[iCol - 1, LocationDigits]};
			{iRow, iCol} -> Part @@ Join[{Matrix1}, Indices[[All, 1]]] Part @@ Join[{Matrix2}, 
			Indices[[All, 2]]], {iRow, L1 L2}, {iCol, L1 L2}]]]]];
(* }}} *)
(* {{{ *) tensorProduct[LocationDigits_, State1_?VectorQ, State2_?VectorQ] := 
	Module[{Index, i, L1, L2}, 
		{L1, L2} = Length /@ {State1, State2};
		Normal[SparseArray[Table[Index = 1 + ExtractDigits[i - 1, LocationDigits]; 
		i -> State1[[Index[[1]]]] State2[[Index[[2]]]], {i, L1 L2}]]]];
(* }}} *)
(* {{{ *) tensorProduct[State1_?VectorQ, State2_?VectorQ] := Flatten[KroneckerProduct[State1, State2]]
(* }}} *)
(* {{{ *) tensorPower[A_, n_] := Nest[KroneckerProduct[A, #] &, A, n - 1]
(* }}} *)
(* {{{ *) OrthonormalBasisContaningVector[psi_?VectorQ] := 
(*El Ri generado es orthogonal a psi y tiene norma menor que uno*)
 Module[{n, ri, i, Ri, F, eks},
  n = Length[psi];
  F = Sum[
     ri = #/(Sqrt[Conjugate[#] . #]) &[
       Table[RandomGaussianComplex[], {i, n}]];
     Ri = ri - (Conjugate[psi] . ri) psi;
     {Conjugate[Ri] . psi, Conjugate[Ri] . Ri};
     Proyector[Ri], {n - 1}] + Proyector[psi];
  eks = Transpose[
     Sort[Transpose[Eigensystem[F]], 
      Abs[#1[[1]] - 1] < Abs[#2[[1]] - 1] &]][[2]];
  Exp[I (Arg[psi[[1]]] - Arg[eks[[1, 1]]])] #/Norm[#] & /@
    eks]
(* }}} *)
(* {{{  *) GetMatrixForm[Gate_, Qubits_] := Transpose[Gate /@ IdentityMatrix[Power[2, Qubits]]]
(* }}} *)
(* }}} *)
(* {{{ Quantum information routines *)
(* {{{ *)ApplyOperator[operator_,rho_] := operator . rho . Dagger[operator]
(* }}} *)
(* {{{ *) (*ACG: ApplyChannel could be written as a composition of Map and ApplyOperator*)
ApplyChannel[Es_, rho_] := 
 Sum[Es[[k]] . rho . Dagger[Es[[k]]], {k, Length[Es]}]
(* }}} *)
(* {{{ *) Hadamard[] := 
 {{1,1},{1,-1}}/Sqrt[2]
(* }}} *)
(* {{{ *) ControlNotMatrix[QubitControl_Integer, QubitTarget_Integer, NumberOfQubits_Integer] :=
 SparseArray[
  Table[{NumeroACambiar, ControlNot[QubitControl, QubitTarget, NumeroACambiar - 1] + 1}, 
  	{NumeroACambiar, Power[2, NumberOfQubits]}] -> 1]
ControlNot[QubitControl_Integer, QubitTarget_Integer, NumeroACambiar_Integer] := 
 Module[{NumeroEnBinario, DigitoControl, DigitoTarget, n},
  NumeroEnBinario = IntegerDigits[NumeroACambiar, 2];
  n = Max[Length[NumeroEnBinario], QubitControl + 1, 
    QubitTarget + 1];
  NumeroEnBinario = IntegerDigits[NumeroACambiar, 2, n];
  DigitoControl = NumeroEnBinario[[-QubitControl - 1]];
  DigitoTarget = NumeroEnBinario[[-QubitTarget - 1]];
  NumeroEnBinario[[-QubitTarget - 1]] = 
   Mod[DigitoControl + DigitoTarget, 2];
  FromDigits[NumeroEnBinario, 2]]
(* }}} *)
(* {{{ *) (*SWAP matrix using sprase arrays and *)
SWAPMatrix[n_Integer, {j_Integer, k_Integer}] /; 
  1 <= j <= n && 1 <= k <= n && j != k := 
 Total[LocalOneQubitOperator[n, j, #] . 
      LocalOneQubitOperator[n, k, #] & /@ 
    Table[Pauli[i], {i, 0, 3}]]/2
SWAPMatrix[n_Integer, {j_Integer, k_Integer}] /; 
  1 <= j <= n && 1 <= k <= n && j == k := IdentityMatrix[2^n, SparseArray]
(* }}} *)
(* {{{ *) QuantumDotProduct[Psi1_?VectorQ, Psi2_?VectorQ] :=  Dot[Conjugate[Psi1], Psi2]
(* }}} *)
(* {{{ *) ToSuperOperatorSpaceComputationalBasis[rho_?MatrixQ] := 
            Flatten[rho]
(* }}} *)
(* {{{ *) FromSuperOperatorSpaceComputationalBasis[rho_?VectorQ] := 
 Partition[rho, Sqrt[Length[rho]]]
(* }}} *)
(* {{{ *) ToSuperOperatorSpacePauliBasis[r_?MatrixQ] := Module[{Qubits},
  Qubits = Log[2, Length[r]];
  Chop[Table[ Tr[r . Pauli[IntegerDigits[i, 4, Qubits]]/Power[2.,Qubits/2]], {i, 0, Power[Length[r], 2] - 1}]]]
(* }}} *)
(* {{{ *) FromSuperOperatorSpacePauliBasis[r_?VectorQ] := Module[{Qubits},
  Qubits = Log[2, Length[r]]/2; 
  Sum[r[[i + 1]] Pauli[IntegerDigits[i, 4, Qubits]]/
	  Power[2., Qubits/2], {i, 0, Length[r] - 1}]
   ]
(* }}} *)
(* {{{  *) BlochNormalFormVectors[r_?MatrixQ] := Module[{R, T, oa, cd, ob, A, a, b, c},
  R = Chop[StateToBlochBasisRepresentation[r]];
  T = R[[2 ;;, 2 ;;]];
  {oa, cd, ob} = SingularValueDecomposition[T];
  A = Chop[
    DirectSum[{{1}}, Transpose[oa/Det[oa]]] . R . DirectSum[{{1}}, 
      ob/Det[ob]]];
  {a = A[[2 ;;, 1]], b = A[[1, 2 ;;]], c = Diagonal[A][[2 ;;]]}
  ]
(* }}} *)
(* {{{  *) BlochNormalFormFromVectors[{a_, b_, c_}] := Module[{i},
  (IdentityMatrix[4] + Sum[b[[i]] Pauli[{0, i}], {i, 1, 3}] + 
   Sum[a[[i]] Pauli[{i, 0}], {i, 1, 3}] + 
   Sum[c[[i]] Pauli[{i, i}], {i, 1, 3}])/4
  ]


(* }}} *)
(* {{{  *) StateToBlochBasisRepresentation[r_?MatrixQ] := Table[Tr[r . Pauli[{i, j}]], {i, 0, 3}, {j, 0, 3}]
(* }}} *)
(* {{{ *) CoherentState[\[Theta]_, \[Phi]_, n_] := 
 N[Flatten[
   tensorPower[{Cos[\[Theta]/2], Exp[I \[Phi]] Sin[\[Theta]/2]}, n], 
   1]]
(* }}} *)
(* {{{ *)LocalOneQubitOperator[n_Integer, k_Integer, a_] /; 1<=k<=n && Dimensions[a]=={2,2} :=
KroneckerProduct[IdentityMatrix[2^(k-1), SparseArray],
a,
IdentityMatrix[2^(n-k), SparseArray]]
(* }}} *)
(* {{{ Pauli matrices*)  
Pauli[i_Integer]:=SparseArray[PauliMatrix[i]]
Pauli[Indices_List] := KroneckerProduct @@ (Pauli /@ Indices)
(* }}} *)
(* {{{ *) Paulib[{b1_,b2_,b3_}] := b . Table[Pauli[i],{i,1,3}]
(* }}} *)
(* {{{ *) SumSigmaX[Qubits_] := SumSigmaX[Qubits] = Table[If[DigitCount[BitXor[i - 1, j - 1], 2, 1] == 1, 1, 0], {i, Power[2, Qubits]}, {j, Power[2, Qubits]}]
(* }}} *)
(* {{{ *) SumSigmaY[Qubits_] := SumSigmaY[Qubits] = Table[ If[DigitCount[BitXor[i - 1, j - 1], 2, 1] == 1, If[i > j, I, -I], 0], {i, Power[2, Qubits]}, {j, Power[2, Qubits]}]
(* }}} *)
(* {{{ *) SumSigmaZ[Qubits_] := SumSigmaZ[Qubits] = DiagonalMatrix[ Table[Qubits - 2 DigitCount[i, 2, 1], {i, 0, Power[2, Qubits] - 1}]]
(* }}} *)
(* {{{ *) ValidDensityMatrix[Rho_?MatrixQ] := (Abs[Tr[Rho] - 1]<Power[10,-13] && 
	Length[Select[# >= 0 & /@ Chop[Eigenvalues[Rho]], ! # &]] == 0)
(* }}} *)
(* {{{ *) Dagger[H_]:=Conjugate[Transpose[H]]

(* }}} *)
(* {{{ *) Concurrence[rho_?MatrixQ]:=Module[{lambda}, 
	lambda=Sqrt[Abs[Eigenvalues[rho . sigmaysigmay . Conjugate[rho] . sigmaysigmay]]]; 
	Max[{2*Max[lambda]-Plus@@lambda,0}]]
Concurrence[\[Psi]_?VectorQ]:=Module[{}, 
Abs[QuantumDotProduct[sigmaysigmay . Conjugate[\[Psi]], \[Psi]]]]

(* {{{ *) sigmaysigmay={{0, 0, 0, -1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {-1, 0, 0, 0}}; (* }}} *)

(* {{{ *) MultipartiteConcurrence[\[Psi]_?VectorQ] := Module[{M},M=Length[\[Psi]]-2;
Sqrt[M-Sum[Purity[PartialTrace[\[Psi],i]],{i,1,M}]]*(2/Sqrt[M+2])](* }}} *)
(* }}} *)
(* {{{ *) Purity[rho_]:= Tr[rho . rho]

(* }}} *)
(* {{{ Proyector *)
Proyector[psi_, phi_] := Outer[Times, psi, Conjugate[phi]]
Proyector[psi_] := Proyector[psi, psi]
(* }}} *)
(* {{{ *) ApplyKrausOperators[Operators_, rho_] := 
 Total[# . rho . Dagger[#] & /@ Operators]
(* }}} *)
(* {{{ *) KrausOperatorsForSuperUnitaryEvolution[psienv_, U_] := 
  Module[{Nenv, Ntotal, Nsub},
   (*
   See Chuang and Nielsen p. 360
   So, We have that E(rho)=sum_k Ek rho Ek(dagger);
   One can see that Ek=<ek|U|psienv>, 
   that is a matrix with matrix elements
   Ek_{ij}=<i|Ek|j> = <i ek|U|psienv j>;
   Metiendo una identidad obrenemos que 
   Ek_{ij}=sum_m <ek i|U|m j > <m|psienv>;
   *)
   Nenv = Length[psienv];
   Ntotal = Length[U];
   Nsub = Ntotal/Nenv;
   Table[(*Print["k="<>ToString[k]];*)
    
    Table[(*Print["i="<>ToString[i]<>" j="<>ToString[j]];*)
     Sum[
      (*If[k==2,Print["k="<>ToString[k]<>" i="<>ToString[i]<>" j="<>
      ToString[j]<>" m="<>ToString[m]<>" m="<>ToString[{Nsub k+i+1,
      Nsub m+j+1,m+1}]],];*)
      
      U[[Nsub k + i + 1, Nsub m + j + 1]] psienv[[m + 1]], {m, 0, 
       Nenv - 1}], {i, 0, Nsub - 1}, {j, 0, Nsub - 1}],
    {k, 0, Nenv - 1}]];
(* }}} *)
(* {{{ *) vonNeumannEntropy[r_?MatrixQ] := vonNeumannEntropy[Eigenvalues[r]]
(* }}} *)
(* {{{ *) vonNeumannEntropy[r_?ListQ] := -Total[If[# == 0, 0, # Log[2, #]] & /@ r]
(* }}} *)
(* {{{ *) Bell[n_] := (1/Sqrt[n]) Table[If[Mod[i - 1, n + 1] == 0, 1, 0], {i, n^2}]
(* }}} *)
(* {{{ *) Bell[] = Bell[2]

(* }}} *)
(* {{{  *) StateToBlochSphere[R_?MatrixQ] := Module[{sigma}, sigma={Pauli[1], Pauli[2], Pauli[3]}; Chop[Tr /@ (sigma . R)]]
(* }}} *)
(* {{{  *) BlochSphereToState[CartesianCoordinatesOfPoint_List] :=  
 Module[{r, \[Theta], \[Phi]}, 
  r = CoordinateTransform["Cartesian"->"Spherical", CartesianCoordinatesOfPoint]; 
	\[Theta] = r[[2]]; \[Phi] = r[[3]]; {Cos[\[Theta]/2], Exp[I \[Phi]] Sin[\[Theta]/2]}]
BlochSphereToState[{\[Theta]_, \[Phi]_}] :=  {Cos[\[Theta]/2], Exp[I \[Phi]] Sin[\[Theta]/2]}
(* }}} *)
(* {{{  *) QuantumMutualInformationReducedEntropies[r_?MatrixQ] := Module[{ra, rb},
  ra = PartialTrace[r, 1]; rb = PartialTrace[r, 2];
  vonNeumannEntropy[ra] + vonNeumannEntropy[rb] -  vonNeumannEntropy[r]]
(* }}} *)
(* {{{  *) QuantumMutualInformationMeasurements[r_?MatrixQ] := 
 Module[{th, phi}, Maximize[ QuantumMutualInformationMeasurements[ r, {th, phi}], {th, phi}][[1]]]


(* {{{  *) QuantumMutualInformationMeasurements[ r_?MatrixQ, {\[Theta]_, \[CurlyPhi]_}] :=
(*Defined in eq 11 of Luo*)
 vonNeumannEntropy[PartialTrace[r, 2]] - 
  ConditionalEntropy[r, {\[Theta], \[CurlyPhi]}]
(* }}} *)
(* {{{  *) ConditionalEntropy[\[Rho]_, {\[Theta]_, \[CurlyPhi]_}] := 
 Module[{X, a, b, c, mm, mp, ps, \[Lambda]s},
  X = {2 Cos[\[Theta]] Sin[\[Theta]] Cos[\[CurlyPhi]], 
    2 Cos[\[Theta]] Sin[\[Theta]] Sin[\[CurlyPhi]], 
    2 Cos[\[Theta]]^2 - 1};
  {a, b, c} = BlochNormalFormVectors[\[Rho]];
  {mp = a + c X, mm = a - c X};
  ps = {(1 + b . X)/2, (1 - b . X)/2};
  \[Lambda]s = {{1/2 (1 + Norm[mp]/(1 + b . X)), 
     1/2 (1 - Norm[mp]/(1 + b . X))}, {1/2 (1 + Norm[mm]/(1 - b . X)), 
     1/2 (1 - Norm[mm]/(1 - b . X))}};
  ps[[1]] vonNeumannEntropy[\[Lambda]s[[1]]] + 
   ps[[2]] vonNeumannEntropy[\[Lambda]s[[2]]]
  ]
(* }}} *)
 
(* }}} *)
(* {{{  *) Discord[r_?MatrixQ] := QuantumMutualInformationReducedEntropies[r] - QuantumMutualInformationMeasurements[r]
(* }}} *)
(* {{{  *) ApplyGate[U_, State_, TargetQubits_] := Module[{StateOut, i, ie},
  StateOut = Table[Null, {Length[State]}];
  For[i = 0, i < Length[State]/2, i++,
   ie = MergeTwoIntegers[#, i, 
       Power[2, TargetQubits]] & /@ (Range[Length[U]] - 1);
   StateOut[[ie + 1]] = U . State[[ie + 1]];
   ];
  StateOut]
(* }}} *)
(* {{{  *) ApplyControlGate[U_, State_, TargetQubit_, ControlQubit_] := 
 Module[{StateOut, iwithcontrol, ie, NormalizedControlQubit},
  StateOut = State;
  If[ControlQubit > TargetQubit, 
   NormalizedControlQubit = ControlQubit - 1, 
   NormalizedControlQubit = ControlQubit];
  For[i = 0, i < Length[State]/4, i++,
   iwithcontrol = MergeTwoIntegers[1, i, Power[2, NormalizedControlQubit]];
   (*Print[{i,iwithcontrol}];*)
   
   ie = MergeTwoIntegers[#, iwithcontrol, 
       Power[2, TargetQubit]] & /@ {0, 1};
   StateOut[[ie + 1]] = U . State[[ie + 1]];
   ];
  StateOut]
(* }}} *)
(* {{{  *) BasisState[BasisNumber_,Dimension_] := Table[If[i == BasisNumber, 1, 0], {i, 0, Dimension - 1}]
(* }}} *)
(* }}} *)
(* {{{ Quantum channels and basis transformations *)
JamiolkowskiStateToOperatorChoi[Rho_?MatrixQ] := Sqrt[Length[Rho]] Reshuffle[Rho]
JamiolkowskiOperatorChoiToState[O_?MatrixQ] := Reshuffle[O]/Sqrt[Length[O]]
TransformationMatrixPauliBasisToComputationalBasis[] := {{1/Sqrt[2], 0, 0, 1/Sqrt[2]}, {0, 1/Sqrt[2], (-I)/Sqrt[2], 0}, {0, 1/Sqrt[2], I/Sqrt[2], 0}, 
 {1/Sqrt[2], 0, 0, -(1/Sqrt[2])}};
(* {{{ *) Reshuffle[Phi_?MatrixQ] := Module[{Dim, mn, MuNu, m, Mu, n, Nu},
   Dim = Sqrt[Length[Phi]];
   Table[ {m, n} = IntegerDigits[mn, Dim, 2];
    {Mu, Nu} = IntegerDigits[MuNu, Dim, 2];
    Phi[[FromDigits[{m, Mu}, Dim] + 1, 
     FromDigits[{n, Nu}, Dim] + 1]], {mn, 0, 
     Dim^2 - 1}, {MuNu , 0, Dim^2 - 1}]];

 (* }}} *)
(* {{{ *) RandomTracePreservingMapChoiBasis[] := Module[{psi},
  psi = Total[
     MapThread[
      tensorProduct, {TwoRandomOrhtogonalStates[8], 
       TwoRandomOrhtogonalStates[2]}]]/Sqrt[2];
  Reshuffle[2 PartialTrace[Proyector[psi], 3]]
  ]
AveragePurityChannelPauliBasis[Lambda_] := Total[Power[Lambda[[All, 1]], 2]]/2 + Power[Norm[Lambda[[2 ;;, 2 ;;]], "Frobenius"], 2]/6

 (* }}} *)
(* {{{ *) BlochEllipsoid[Cha_]:=Module[{center,T,coord,vecs,x,y,z,vecs0},
  T=(#+DiagonalMatrix[If[#==0,0.01,0]&/@Diagonal[#]])&[Cha[[{2,3,4},{2,3,4}]]];
  center={Cha[[2,1]],Cha[[3,1]],Cha[[4,1]]};
  vecs0=Graphics3D[{{
        Dashing[0.01],Opacity[0.5],Red, Arrow[{{0,0,0},1.3*Normalize[{1,0,0}]}],
        {Dashing[0.01],Opacity[0.5],Blue,Arrow[{{0,0,0},1.3*Normalize[{0,1,0}]}]},
        {Dashing[0.01],Opacity[0.5],Green,Arrow[{{0,0,0},1.3*Normalize[{0,0,1}]}]}
        } }];
  vecs=Graphics3D[{{Red,Arrow[{center,1.3*Normalize[T . {1,0,0}]}],
           {Blue,Arrow[{center,1.3*Normalize[T . {0,1,0}]}]},
           {Green,Arrow[{center,1.3*Normalize[T . {0,0,1}]}]} }}];
  coord={x,y,z}-center;
  coord=Inverse[T] . coord;
  Style[Show[ContourPlot3D[
    {coord[[1]]^2+coord[[2]]^2+coord[[3]]^2==1,x^2+y^2+z^2==1},
    {x,-1,1},{y,-1,1},{z,-1,1},AxesLabel->{"X","Y","Z"},
    ContourStyle->{Automatic,Opacity[0.3]},Mesh->None],vecs,vecs0,PlotRange->1.3],RenderingOptions->{"3DRenderingMethod"->"HardwareDepthPeeling"}]
]
 (* }}} *)
(* {{{ *) EvolvGate[Gate_, steps_, env_, state_]:=
 Module[{statefinal, list, gate, j},
  statefinal = tensorProduct[env,state];
  gate[statefinal_] := First[Gate & /@ {statefinal}];
  list = Table[statefinal = gate[statefinal];
    PartialTrace[statefinal, 1], {j, steps}
    ];
  list
  ]
(*}}}*)
(* {{{ *) MakeQuantumChannel[Gate_, steps_, env_] := 
 Module[{\[Sigma], channel, Y, X, cero, uno},
  Y = EvolvGate[Gate, steps, env, {-I, 1}];
  cero = EvolvGate[Gate, steps, env, {0, 1}];
  uno = EvolvGate[Gate, steps, env, {1, 0}];
  X = EvolvGate[Gate, steps, env, {1, 1}];
  \[Sigma][2] = Y - cero - uno;
  \[Sigma][1] = X - cero - uno;
  \[Sigma][0] = cero + uno;
  \[Sigma][3] = uno - cero;
  Table[channel[i, j] = 
    1/2 Table[Tr[PauliMatrix[i] . \[Sigma][j][[k]]], {k, steps}], {i, 0,
     3}, {j, 0, 3}];
  Chop[Table[
    Table[channel[i, j][[k]], {i, 0, 3}, {j, 0, 3}], {k, steps}]]]
(* }}} *)
(* {{{ *)SumPositiveDerivatives[list_]:=Module[{sum},
sum=0;
Table[
If[list[[i+1]]>list[[i]],
sum=sum+list[[i+1]]-list[[i]];
],
{i,1,Length[list]-1}];
sum]
(* }}} *)
(* {{{  GHZ *)
GHZ[qu_]:=1/Sqrt[2]Table[If[i==0||i==2^qu-1,1,0],{i,0,2^qu-1}]
(* }}} *)
(* {{{  W state *)
Wstate[n_] := Sum[BasisState[Power[2, m], Power[2, n]], {m, 0, n - 1}]/Sqrt[n]
(* }}} *)
(* }}} *)
(*{{{*)ArbitratyPartialTrace[hilbertdimrem_, hilbertdimkeep_, M_] := 
 If[IntegerQ[hilbertdimrem] == True && IntegerQ[hilbertdimkeep] == True,
  \[CapitalXi] = {};
  \[Xi] = {};
  For[s = 0, s < hilbertdimkeep, s++,
   \[Xi] = {};
   For[j = 0, j < hilbertdimkeep, j++,
    AppendTo[\[Xi], 
     Chop[Sum[
       M[[i + s*hilbertdimrem, i + j*hilbertdimrem]], {i, 1, 
        hilbertdimrem}]]]
    ];
   AppendTo[\[CapitalXi], \[Xi]];
   ]; Return[\[CapitalXi]]](*}}}*)

(*{{{ quien sabe quien piso estas aca*)RandomMixedState[n_,combinations_]:=Module[{p,statelist},
statelist=Table[Proyector[RandomState[n]],{combinations}];
p=RandomReal[{0,1.0},combinations];
p=p/Total[p];
p . statelist//Chop
];


GellMann[n_] :=
 GellMann[n] = Flatten[Table[
   (* Symmetric case *)
   SparseArray[{{ j, k} -> 1, { k, j} -> 1}, {n, n}]
  , {k, 2, n}, {j, 1, k - 1}], 1]~
  Join~Flatten[Table[
   (* Antisymmetric case *)
   SparseArray[{{ j, k} -> -I, { k, j} -> +I}, {n, n}]
  , {k, 2, n}, {j, 1, k - 1}], 1]~
  Join~Table[
   (* Diagonal case *)
   Sqrt[2/l/(l + 1)] SparseArray[
    Table[{j, j} -> 1, {j, 1, l}]~Join~{{l + 1, l + 1} -> -l}, {n, n}]
  , {l, 1, n - 1}];




ApplySwap[rho_?VectorQ,j_Integer,k_Integer]:=With[{n = Log2[Length[rho]]},ApplyOperator[SWAPMatrix[n,{j,k}],rho] ]

ApplySwap[rho_?SquareMatrixQ,j_Integer,k_Integer]:=With[{n = Log2[Dimensions[rho][[1]]]},ApplyOperator[SWAPMatrix[n,{j,k}],rho] ]


ApplySwapPure[State_?VectorQ,Target1_,Target2_]:=Module[{Aux,digits,len,digits1,digits2},
len=Length[State];
Aux=ConstantArray[0,len];
Table[digits=IntegerDigits[i-1,2,IntegerPart[Log2[len]]];
digits1=digits;
digits1[[{Target1,Target2}]]=digits[[{Target2,Target1}]];
Aux[[i]]=State[[FromDigits[digits1,2]+1]];,{i,1,Length[State]}];
Aux
];
ApplyPermutation[state_,permmatrix_]:=Module[{State,Aux,digitsinit,digitsfinal,len},
len=Length[state];
Aux=ConstantArray[0,len];
Table[digitsinit=IntegerDigits[i-1,2,IntegerPart[Log2[len]]];
digitsfinal=permmatrix . digitsinit;
Aux[[i]]=state[[FromDigits[digitsfinal,2]+1]];,{i,1,Length[state]}];
Aux
]

(*}}}*)

(*Coarse Graining stuff*) (*{{{*)
ApplyLocalNoiseChain[State_?MatrixQ,p_]:=Module[{qubits},
qubits=IntegerPart[Log2[Dimensions[State][[1]]]];
p State+(1-p)/(qubits)(Sum[ApplySwap[State,i,Mod[i+1,qubits,1]],{i,1,qubits}])
];
ApplyNoiseChain[State_?MatrixQ,p_]:=Module[{qubits},
qubits=IntegerPart[Log2[Dimensions[State][[1]]]];
p State+2(1-p)/(qubits(qubits-1))(Sum[ApplySwap[State,i,j],{i,1,qubits},{j,i+1,qubits}])
];
ApplyLocalNoiseChain[State_?VectorQ,p_]:=Module[{qubits},
qubits=IntegerPart[Log2[Dimensions[State][[1]]]];
p Proyector[State]+(1-p)/(qubits)(Sum[ApplySwap[State,i,Mod[i+1,qubits,1]],{i,1,qubits}])
];
ApplyNoiseChain[State_?VectorQ,p_]:=Module[{qubits},
qubits=IntegerPart[Log2[Dimensions[State][[1]]]];
p Proyector[State]+2(1-p)/(qubits(qubits-1))(Sum[ApplySwap[State,i,j],{i,1,qubits},{j,i+1,qubits}])
];
(*
Commented as is provided by Mathematica in 13.1
PermutationMatrix[p_List]:=IdentityMatrix[Length[p]][[p]];
*)
PermutationMatrices[n_]:=PermutationMatrix/@Permutations[Range[n]];
(*PCE operations related stuff.*)
PauliSuperoperator[pauliDiagonal_List]:=Module[{n,pauliToComputational,diagonal},
diagonal=pauliDiagonal//Flatten;
n=Log[4,Length[diagonal]];
pauliToComputational=tensorPower[TransformationMatrixPauliBasisToComputationalBasis[],n];
pauliToComputational . DiagonalMatrix[diagonal] . Inverse[pauliToComputational]
];
(*JA: PCEFigures est\[AAcute] programada del asco. Un d\[IAcute]a con ganas de distraerme la arreglo. Para mientras hace la tarea.*)
PCEFigures[correlations_]:=Module[{cubeIndices,diagonalPCE},
cubeIndices=Position[correlations,1]-1;
If[Length[Dimensions[correlations]]==3,
Graphics3D[{If[Count[#,0]==3,{Black,Cube[#]},
If[Count[#,0]==2,{RGBColor["#CC0000"],Cube[#]},
If[Count[#,0]==1,{RGBColor["#004C99"],Cube[#]},
If[Count[#,0]==0,{RGBColor["#99FF33"],Cube[#]}]]]]&/@cubeIndices,
{Thickness[0.012],Line[{{{-0.5,-0.5,-0.5},{-0.5,-0.5,3.5}},{{-0.5,-0.5,-0.5},{-0.5,3.5,-0.5}},{{-0.5,-0.5,-0.5},{3.5,-0.5,-0.5}},
{{3.5,-0.5,-0.5},{3.5,-0.5,3.5}},
{{-0.5,-0.5,3.5},{3.5,-0.5,3.5}},
{{-0.5,3.5,-0.5},{3.5,3.5,-0.5}},
{{3.5,3.5,-0.5},{3.5,3.5,3.5}},
{{3.5,3.5,3.5},{-0.5,3.5,3.5}},
{{-0.5,3.5,3.5},{-0.5,3.5,-0.5}},
{{-0.5,3.5,3.5},{-0.5,-0.5,3.5}},
{{3.5,3.5,3.5},{3.5,-0.5,3.5}},
{{3.5,3.5,-0.5},{3.5,-0.5,-0.5}}}]}},
Axes->False,AxesLabel->{"x","y","z"},LabelStyle->Directive[Bold,Medium,Black],PlotRange->{{-0.5,3.5},{-0.5,3.5},{-0.5,3.5}},AxesOrigin->{0.5,0.5,0.5},AxesStyle->Thickness[0.005],ImageSize->Medium,ImagePadding->45],
If[Length[Dimensions[correlations]]==2,
diagonalPCE=correlations//Flatten;
ArrayPlot[SparseArray[Position[ArrayReshape[diagonalPCE,{4,4}],1]->(If[#[[1]]==1\[And]#[[2]]==1,Black,If[#[[1]]==1\[Or]#[[2]]==1,RGBColor["#CC0000"],If[#[[1]]!=1\[And]#[[2]]!=1,RGBColor["#004C99"],Nothing]]]&/@Position[ArrayReshape[diagonalPCE,{4,4}],1]),{4,4}]]
]
]
];

ApplyMultiQubitGate::dimerr="Invalid target dimensions `1`. The condition 2^Total[IntegerDigits[targets, 2]] == dimtarget must hold.";
ApplyMultiQubitGate[state_?VectorQ,gate_,targets_]:=Module[{statenew,pos,dimtarget,dimtotal,tmp},
statenew=state;
dimtarget=Length[gate];
dimtotal=Length[state];
If[2^Total[IntegerDigits[targets,2]]=!=dimtarget,Message[ApplyMultiQubitGate::dimerr,targets];
Return[$Failed];];
Table[
pos=Table[MergeTwoIntegers[targetindex,untouched,targets],{targetindex,0,dimtarget-1}]+1;
statenew[[pos]]=gate . state[[pos]];
,{untouched,0,dimtotal/dimtarget-1}];
statenew//Chop
];

ApplyMultiQubitGate[state_?MatrixQ,gate_,targets_]:=Module[{statenew,pos,pos2,dimtarget,dimtotal,tmp},
statenew=state;
dimtarget=Length[gate];
dimtotal=Length[state];
If[2^Total[IntegerDigits[targets,2]]=!=dimtarget,Message[ApplyMultiQubitGate::dimerr,targets];
Return[$Failed];];
Table[
pos=Table[MergeTwoIntegers[targetindex,untouched,targets],{targetindex,0,dimtarget-1}]+1;
pos2=Table[MergeTwoIntegers[targetindex,untouched2,targets],{targetindex,0,dimtarget-1}]+1;
statenew[[pos,pos2]]=gate . state[[pos,pos2]] . Dagger[gate];
,{untouched,0,dimtotal/dimtarget-1},
{untouched2,0,dimtotal/dimtarget-1}];
statenew//Chop
];
(*}}}*)
(*}}}*)
End[] 
EndPackage[]
(* }}} *)





(* --- End: libsCP/Quantum.m --- *)



(* --- Begin: libsCP/SpinChain.m --- *)


(* ::Package:: *)

(* {{{ *) BeginPackage["SpinChain`",{"Carlos`", "Quantum`"}]
(* {{{ Primitives *)
ApplyMagnetickKick::usage = "ApplyMagnetickKick[state_, b_, Target_] or ApplyMagnetickKick[state_, b_]"
ApplyIsing::usage = "ApplyIsing[state_, J_, Position1_, Position2_]"
(* }}} *)
(* {{{ Full topologies*)
ApplyIsingTorus::usage="Se hace la topologia de un toro. Solo la parte de Ising"
ApplyIsingChain::usage="Se hace la topologia de una cadena. Solo la parte de Ising"
ApplyChain::usage="Se hace la topologia de una cadena."
ApplyInverseChain::usage="Se hace la topologia de una cadena pero hacia atras en el tiempo."
ApplyCommonEnvironment::usage="Se refiere a la topologia (a) del PRL de n-body Bell en PRA."
ApplyDephasingChain::usage="ApplyDephasingChain[psi0_, Delta_, Jenv_, benv_, Jinteraction_] Se hace la topologia de una cadena que solo hace dephasing"
ApplyChainStar::usage="ApplyChainStar[state_, Jenv_,Jint_, b_] Se hace la topologia de la estrella con el magnetic kick"
ApplyIsingStar::usage="Se hace la topologia de una estrella, solo la parte de Ising"
ApplyIsingStarEnvironment::usage="Se hace la topologia de una estrella, solo la parte de Ising, en el environment"
ApplyMagnetickKickStarEnvironment::usage="Se hace el kick pero solo en los qubits que son del environment"
ApplyIsingStarInteractionQubitEnvironment::usage="Se hace el kick pero solo la interaccion con el  environment"
ApplyInomogeneousChain::usage="ApplyInomogeneousChain[state_?VectorQ, J_, J10_, b_] Se hace la cadena inhomogenea donde J10 indica la interaccion ising entre el primero y segundo qubit"
(* }}} *)
(* {{{ Explicit Matrices *)
IsingMatrix::usage="Get the matrix for the Ising Interaction Sigma_i Sigma_j, or for sum Sigma_i Sigma_j. In the first case, call as IsingMatrix[{IsingPosition1_Integer, IsingPosition2_Integer}, Qubits_Integer], and in the second,  IsingMatrix[IsingPositions_List, Qubits_Integer]"
SpinChainIsingMatrix::usage="The Ising matrix that has to come in the IsingMatrix routine, for a perdic spin chain. Call as SpinChainIsingMatrix[Qubits_]  "
SpinGridIsingMatrix::usage="The Ising matrix that has to come in the IsingMatrix routine, for a toric spin grid. Call as SpinGridIsingMatrix[{nx_, ny_}]"
MatrixPauliMagneticField::usage="Matrix corresponding to the hamiltonian b.sum sigma_j"
HamiltonianMagenitcChain::usage="Matrix Corresponding to the continuous Periodic ising spin chain with magnetic field"
HamiltonianMagenitcGrid::usage="Matrix Corresponding to the continuous toric ising grid chain with magnetic field"
ApplyMagnetickKickInhom::usage="ApplyMagnetickKickInhom[state_, b_, binhom_], Se aplica el magnetick kick donde binhom es el campo del spin 0"
ApplyIsingAllVsAll::usage"ApplyIsingAllVsAll[state_,J_]"
(* }}} *)
(* }}} *)
Begin["Private`"] 
(* {{{ Primitives*)
(* {{{ *) ApplyMagnetickKick[state_?VectorQ, b_, Target_] := 
 Module[{RotationMatrix, statenew, i, pos},
  RotationMatrix = MatrixExp[-I MatrixPauliMagneticField[b,1]];
  statenew = state;
  For[i = 0, i < Length[state]/2, i++,
   pos = {MergeTwoIntegers[0, i, Power[2, Target]], 
      MergeTwoIntegers[1, i, Power[2, Target]]} + 1;
   statenew[[pos]] = RotationMatrix.statenew[[pos]];];
  statenew]
(* }}} *)
(* {{{ *) ApplyMagnetickKick[state_?VectorQ, b_] := 
 Module[{Qubits, statenew, QubitToAdress},
  Qubits = Log[2, Length[state]];
  statenew = state;
  For[QubitToAdress = 0, QubitToAdress < Qubits, QubitToAdress++, 
   (*Print["en el loop QubitToAdress="<>ToString[QubitToAdress]];*)
   statenew = ApplyMagnetickKick[statenew, b, QubitToAdress]];
  statenew]
(* }}} *)
(* {{{ *) ApplyMagnetickKickStarEnvironment[state_?VectorQ, b_] := 
 Module[{Qubits, statenew, QubitToAdress},
  Qubits = Log[2, Length[state]];
  statenew = state;
  For[QubitToAdress = 1, QubitToAdress < Qubits, QubitToAdress++, 
   (*Print["en el loop QubitToAdress="<>ToString[QubitToAdress]];*)
   statenew = ApplyMagnetickKick[statenew, b, QubitToAdress]];
  statenew]
(* }}} *)
(* {{{ *) ApplyIsing[state_?VectorQ, J_, Position1_, Position2_] := 
 Module[{scalar, i, statenew},
  scalar = Exp[-I J];
  statenew = state;
  For[i = 0, i < Length[state], i++,
   If[BitGet[i, Position1] == BitGet[i, Position2], 
    statenew[[i + 1]] = scalar statenew[[i + 1]], 
    statenew[[i + 1]] = Conjugate[scalar] statenew[[i + 1]]]];
  statenew]
(* }}} *)
(* }}} *)
(* {{{ Full topologies *)
(* {{{ *) ApplyIsingChain[state_?VectorQ, J_] := Module[{Qubits, statenew, QubitToAdress,q},
  Qubits = Log[2, Length[state]];
  statenew=state;
  For[q=0, q<Qubits-1, q++, 
    statenew = ApplyIsing[statenew, J, q, q+1];
  ];
  statenew = ApplyIsing[statenew, J, 0 , Qubits-1];
  statenew]
(* }}} *)
(* {{{ *) ApplyInverseChain[state_?VectorQ, J_, b_] := Module[{statenew},
 statenew = state;
 statenew = ApplyMagnetickKick[statenew, -b];
 statenew = ApplyIsingChain[statenew, -J];
 statenew ]
(* }}} *)
(* {{{ *) ApplyChain[state_?VectorQ, J_, b_] := Module[{statenew},
 statenew = state;
 statenew = ApplyIsingChain[statenew, J];
 statenew = ApplyMagnetickKick[statenew, b];
 statenew ]
(* }}} *)
(* {{{ *) ApplyCommonEnvironment[state_?VectorQ, b_, Jenv_, Jcoupling_] := 
 Module[{statenew, J, qubits}, qubits = Log[2, Length[state]];
  J = Table[Jenv, {qubits}];
  J[[1]] = 0;
  J[[2]] = Jcoupling;
  J[[-1]] = Jcoupling;
  ApplyIsing[ApplyMagnetickKick[state, b], J]]
(* }}} *)
(* {{{ *) ApplyDephasingChain[psi0_, Delta_, Jenv_, benv_, Jinteraction_] := Module[{statenew},
  statenew = psi0;
 (*U2 interno del env, las ising y la interaccion con el medio*)
 statenew = ApplyIsingStar[statenew, Jenv, Jinteraction];
 (* En el qubit solo se hace en direccion z, en el resto de la cadena
    donde sea, para que sea integrable/caotica. 
 *)
 statenew = ApplyMagnetickKickStarEnvironment[statenew, benv];
 statenew = ApplyMagnetickKick[statenew, {0, 0, Delta/2}, 0];
 statenew]
(* }}} *)
(* {{{ *) ApplyChainStar[state_?VectorQ, Jenv_,Jint_, b_] := Module[{statenew},
 statenew = state;
 statenew = ApplyIsingStar[statenew, Jenv, Jint];
 statenew = ApplyMagnetickKick[statenew, b];
 statenew ]
(* }}} *)
(* {{{ *) ApplyIsingStar[state_?VectorQ, Jenv_, Jint_] := Module[{Qubits, statenew, QubitToAdress, q},
  Qubits = Log[2, Length[state]];
If[IntegerQ[Qubits]==False,Print["Error: The state does not correspond to a integer number of qubits"];Abort[]];
  statenew=state;
  statenew = ApplyIsingStarEnvironment[statenew, Jenv];
  statenew = ApplyIsingStarInteractionQubitEnvironment[statenew, Jint];
  statenew
  ];
(* }}} *)
(* {{{ *) ApplyIsingStarEnvironment[state_?VectorQ, Jenv_] := Module[{Qubits, statenew, QubitToAdress, q},
  Qubits = Log[2, Length[state]];
If[IntegerQ[Qubits]==False,Print["Error: The state does not correspond to a integer number of qubits"];Abort[]];
  statenew=state;
  For[q=1, q<Qubits-1, q++, statenew = ApplyIsing[statenew, Jenv, q , q+1]; ];
  statenew = ApplyIsing[statenew, Jenv, Qubits-1 , 1]; 
  statenew
  ];
(* }}} *)
(* {{{ *) ApplyIsingStarInteractionQubitEnvironment[state_?VectorQ, Jint_] := Module[{Qubits, statenew, QubitToAdress, q},
  Qubits = Log[2, Length[state]];
If[IntegerQ[Qubits]==False,Print["Error: The state does not correspond to a integer number of qubits"];Abort[]];
  statenew=state;
  For[q=1, q<Qubits, q++, statenew = ApplyIsing[statenew, Jint, 0 , q]; ];
  statenew
  ];
(* }}} *)
(* {{{ *) ApplyInomogeneousChain[state_?VectorQ, J_, J10_, b_] := Module[{Qubits, statenew, QubitToAdress, q},
  Qubits = Log[2, Length[state]];
If[IntegerQ[Qubits]==False,Print["Error: The state does not correspond to a integer number of qubits"];Abort[]];
  statenew=state;
statenew=ApplyIsing[statenew, J10, 0, 1];
  For[q=1, q<Qubits-1, q++, 
	statenew = ApplyIsing[statenew, J, q , q + 1]; 
	];
  statenew = ApplyIsing[statenew, J, 0 , Qubits-1];
	statenew = ApplyMagnetickKick[statenew, b];
  statenew
  ];
(* }}} *)
(* {{{ *) ApplyMagnetickKickInhom[state_, b_, binhom_] := Module[{finalstate, Qubits, i},
 Qubits = Log[2, Length[state]];
  finalstate = ApplyMagnetickKick[state, binhom, 0];
  For[i = 1, i < Qubits, i++, 
   finalstate = ApplyMagnetickKick[finalstate, b, i]];
  finalstate]
(* }}} *)
(* {{{ *)ApplyIsingAllVsAll[state_,J_]:=Module[{statenew, Qubits, i, j},
 Qubits = Log[2, Length[state]];
  statenew = state;
For[i=0,i<Qubits,i++,
For[j=1+i, j<Qubits,j++,
statenew=ApplyIsing[statenew,J,i,j];
]];
statenew]
(* }}} *)
(* }}} *)
(* {{{ Explicit Matrices *)
IsingMatrix[{IsingPosition1_Integer, IsingPosition2_Integer}, Qubits_Integer] := Pauli[Table[ If[l == Qubits - IsingPosition1 || l == Qubits - IsingPosition2, 3, 0], {l, Qubits}]];
IsingMatrix[IsingPositions_List, Qubits_Integer] := Module[{IsingPosition}, Sum[IsingMatrix[IsingPosition, Qubits], {IsingPosition, IsingPositions}]]
SpinChainIsingMatrix[Qubits_] := Table[{Mod[i, Qubits], Mod[i + 1, Qubits]}, {i, 0, Qubits - 1}]

SpinGridIsingMatrix[{nx_, ny_}] := Join[Table[{i, i + 1 + If[Mod[i, nx] == nx - 1, -nx, 0]}, {i, 0, nx ny - 1}], Table[{i, Mod[i + nx, nx ny]}, {i, 0, nx ny - 1}]]

MatrixPauliMagneticField[MagneticField_, Qubits_] := MagneticField.{SumSigmaX[Qubits], SumSigmaY[Qubits], SumSigmaZ[Qubits]}
HamiltonianMagenitcChain[MagneticField_, J_, Qubits_] := MatrixPauliMagneticField[MagneticField, Qubits] + J IsingMatrix[SpinChainIsingMatrix[Qubits], Qubits]
HamiltonianMagenitcGrid[MagneticField_, J_, {nx_, ny_}] := MatrixPauliMagneticField[MagneticField, nx ny] + J IsingMatrix[SpinGridIsingMatrix[{nx, ny}]]
(* }}} *)
End[] 
EndPackage[]




(* --- End: libsCP/SpinChain.m --- *)



(* --- Begin: libsCP/ClassicalMaps.m --- *)


(* {{{ *) BeginPackage["ClassicalMaps`"]
(* {{{ Maps *)
sawTooth::usage="sawTooth[{q_, p_}, K_] gives the sawtooth as in DOI:	10.1016/S0375-9601(97)00455-6"
harper::usage="Harper map, as in our article with Diego Wisniacki, usage: harper[{q_, p_}, {k_, kp_}], harper[{q_, p_}, k_]"
standardMap::usage="Standard Map, [0,1]x[0,1] standardMap[{q_, p_}, K_]"
standardMapNonPeriodic::usage="Standard Map, without periodic conditions, [0,1]x[0,1] standardMapNonPeriodic[{q_, p_}, K_]"
(* }}} *)
Begin["`Private`"] 
(* {{{ Maps *)
(* {{{ *) sawTooth[{q_, p_}, K_] := Module[{qp, pp},
  pp = p + K (Mod[q, 1] - .5);
  qp = q + pp;
  {qp, pp}] 
(* }}} *)
(* {{{ *) standardMap[{q_, p_}, K_] :=
	{Mod[#[[1]],1],#[[2]]}&[standardMapNonPeriodic[{q, p}, K]]
(* }}} *)
(* {{{ *) standardMapNonPeriodic[{q_, p_}, K_] := 
	Module[{qp, pp}, 
		pp = p + K (Sin[2 \[Pi] q])/(2 \[Pi]);
  		qp = q + pp;
 	 {qp, pp}]



(* }}} *)
(* {{{ *) harper[{q_, p_}, {k_, kp_}] := Module[{qp, pp},
  pp = p - k Sin[2. \[Pi] q];
  qp = q + kp Sin[2. \[Pi] pp];
  {qp, pp}]
(* }}} *) 
(* {{{ *) harper[{q_, p_}, k_] := harper[{q, p}, {k, k}]
(* }}} *)
(* {{{ *) 
(* }}} *)
(* }}} *)
End[] 
EndPackage[] (* }}} *)




(* --- End: libsCP/ClassicalMaps.m --- *)



(* --- Begin: libsCP/RankEvolution.m --- *)


BeginPackage["RankEvolution`"]

Needs["Carlos`"]
(*
Needs["Quantum`"]
*)
(* Model simulation {{{*)
AlgorithStepNormal::usage = "Evolves a system one time step acording to the gaussian scale invariant model. Usage, AlgorithStepNormal[WordOrder_, \[Alpha]_]"
Model::usage = "Gives a full model with arbitrary time steps. Usage, Model[NumberOfWords_, TimeSteps_, alpha_]"
Diversity::usage = "Gives the diversity for a full model. The model is a matrix containing the data per year. Usage: Diversity[data_]"
ProbabilityOfChange::usage = "Gives the probability of change of a Model. Usage: ProbabilityOfChange[data_]"
ListToOrdered::usage = "Gives an unsorted and with any kind of data, a list of nombers that reflects the order"
GenerateMixedModel::usage = "Generates the model that interpolates between the closed and the open one. Usage: GenerateMixedModel[probability_, LengthModel_, EvolutionTime_] or GenerateMixedModel[probability_, LengthModel_, EvolutionTime_, skip_]"	
ChangePosition::usage = "Changes the position according to the null model that is written. Usage, ChangePosition[Psi0_, {a_Integer, b_Integer}], ChangePosition[Psi0_] or ChangePosition[Psi0_, skip_Integer]"
FastProbabilityOfDisplacement::usage = "Calculates the probability of displacement, in a fast way. Usage FastProbabilityOfDisplacement[k_, n_, s_, i_]"
GenerateModel2::usage = "Generates the null model"
ModelStepIonizing::usage = "One step of the ionizing model"
GenerateModelStepIonizing::usage = "Complete set of steps for the ionizing model"
(*}}}*)
Begin["`Private`"]
(* Ionizing Model EL BUENO {{{*)
ModelStepIonizing[ ModelState_List, {ProbabilityLevy_, ProbabilityReplace_}] :=
 Module[{Psi, LatestAddedState},
  {Psi, LatestAddedState} = ModelState;
  Psi = If[Random[] < ProbabilityLevy, ChangePosition[Psi], Psi]; 
  If[Random[] < ProbabilityReplace,
   LatestAddedState = LatestAddedState + 1;
   Psi = ReplacePart[Psi, 
     RandomInteger[{1, Length[Psi]}] -> LatestAddedState];];
  {Psi, LatestAddedState}]

ModelStepIonizing[LengthModel_Integer, ProbabilitySet_List, EvolutionTime_] := 
 Nest[ModelStepIonizing[#, ProbabilitySet] &, {Range[LengthModel], 
    LengthModel}, EvolutionTime][[1]]

ModelStepIonizing[LengthModel_Integer, ProbabilitySet_List] :=
  ModelStepIonizing[LengthModel, ProbabilitySet,1]

GeneratePreModelIonizing[LengthModel_, EvolutionTime_, ProbabilitySet_] :=
 NestList[ModelStepIonizing[#, ProbabilitySet] &, {Range[LengthModel],
    LengthModel}, EvolutionTime - 1]

GenerateModelStepIonizing[LengthModel_, EvolutionTime_, ProbabilitySet_] := 
  Module[{completo},
   completo = GeneratePreModelIonizing[LengthModel, EvolutionTime, ProbabilitySet];
   Transpose[completo][[1]][[All, ;; Length[completo[[1, 1]]]]]];

(* Ionizing Model }}}*)
(* Null Model {{{*)
ChangePosition[Psi0_, {a_Integer, b_Integer}] := Flatten[Which[
   a < b, {Psi0[[1 ;; a - 1]], Psi0[[a + 1 ;; b]], Psi0[[a]], Psi0[[b + 1 ;;]]}, 
   a == b, Psi0, 
   a > b, {Psi0[[1 ;; b - 1]], Psi0[[a]], Psi0[[b ;; a - 1]], Psi0[[a + 1 ;;]]}]]
ChangePosition[Psi0_] := ChangePosition[Psi0, RandomInteger[{1, Length[Psi0]}, {2}]]
ChangePosition[Psi0_, skip_Integer] := Nest[ChangePosition, Psi0, skip]

GenerateModel2[LengthModel_, EvolutionTime_, skip_] := 
 NestList[ChangePosition[#, skip] &, Range[LengthModel], EvolutionTime - 1]
GenerateModel2[LengthModel_, EvolutionTime_] := GenerateModel2[LengthModel, EvolutionTime, 1]

ChangePositionToLast[Psi0_, a_Integer] := ChangePosition[Psi0, {a, Length[Psi0]}]
ChangePositionToLast[Psi0_] := ChangePositionToLast[Psi0, RandomInteger[{1, Length[Psi0]}]]

MixedModelIteration[Psi0_, probability_] := If[Random[] <= probability, ChangePosition[Psi0], ChangePositionToLast[Psi0]]
MixedModelIteration[Psi0_, probability_, skip_Integer] := Nest[MixedModelIteration[#, probability] &, Psi0, skip]
GenerateMixedModel[probability_, LengthModel_, EvolutionTime_, skip_] := NestList[MixedModelIteration[#, probability, skip] &, Range[LengthModel], EvolutionTime - 1]
GenerateMixedModel[probability_, LengthModel_, EvolutionTime_] := GenerateMixedModel[probability, LengthModel, EvolutionTime, 1]

(*}}}*)
(* Model simulation {{{*)
AlgorithStepNormal[WordOrder_, \[Alpha]_] := Module[{NumberOfWords, NewPositions},
  NumberOfWords = Length[WordOrder];
  NewPositions = # + RandomReal[NormalDistribution[0, # \[Alpha]]] & /@ Range[NumberOfWords];
  WordOrder[[Ordering[NewPositions]]]]
Model[NumberOfWords_, TimeSteps_, alpha_] := NestList[AlgorithStepNormal[#, alpha] &, Range[NumberOfWords], TimeSteps]
Unfold[lp2_] := Permute[Range[Length[lp2]], InversePermutation[FindPermutation[lp2, Sort[lp2]]]]
(*}}}*)
(* exact formulae {{{*)
(* En al siguiente:

i: Numero de iteraciones
*)
FastProbabilityOfDisplacement[k_, n_, s_, i_] := If[Abs[s] > i, ProbabilidadLevy[i, n], ConvolutionProbabilityOfDisplacement[k, n, s, i]]

ConvolutionProbabilityOfDisplacement[k_, n_, s_, 1] := ProbabilityOfDisplacement[k, n, s, 1]

ConvolutionProbabilityOfDisplacement[k_, n_, s_, i_] := ConvolutionProbabilityOfDisplacement[k, n, s, i] = 
  N[1/n^2 + Sum[FastProbabilityOfDisplacement[k, n, s - a, i - 1] (ProbabilityOfDisplacement[k + s - a, n, a, 1] - 1./n^2), {a, -1, 1}]]

ProbabilidadLevy[I_, n_] := 1./n (1 - (1 - 1/n)^I)

(*FastProbabilityOfDisplacementCorrectRange[k_, n_, s_, i_] := If[1 <= k <= n && 1 <= (s - k) <= n, FastProbabilityOfDisplacement[k, n, s, i], 0]*)

ProbabilityOfDisplacement[CurrentRankK_, LengthModel_, Displacement_, 1] := ProbabilityOfDisplacement[CurrentRankK, LengthModel, Displacement]

ProbabilityOfDisplacement[CurrentRankK_, LengthModel_, Displacement_] := 
 Levy[CurrentRankK, LengthModel, Displacement] + Drift[CurrentRankK, LengthModel, Displacement]

Levy[k_, n_, s_] := If[1 <= k + s <= n, 1/n^2, 0]
Drift[k_, n_, 1] := (n - k) k/n^2
Drift[k_, n_, -1] := (n - k + 1) (k - 1)/n^2
Drift[k_, n_, 0] := (k - 1)^2/n^2 + (n - k)^2/n^2
Drift[k_, n_, s_] := 0
(*}}}*)
Diversity[data_] := NumberList[N[Length[Union[#]]/Length[#]] & /@ Transpose[data]]
NumberOfChanges[x_List] := Length[Split[x]] - 1
ProbabilityOfChangeSingleData[x_List] := NumberOfChanges[x]/(Length[x] - 1)
ProbabilityOfChange[data_] :=NumberList[ProbabilityOfChangeSingleData /@ Transpose[data]]
ListToOrdered[x_] := x /. (Rule @@ # & /@ Transpose[{Sort[x], Range[Length[x]]}])
End[ ]
EndPackage[ ]





(* --- End: libsCP/RankEvolution.m --- *)



(* --- Begin: libsJA/QuantumWalks.wl --- *)


(* ::Package:: *)

(* If ForScience paclet not installed, install it. See https://github.com/MMA-ForScience/ForScience *)
If[Length[PacletFind["ForScience"]]==0, PacletInstall[FileNameJoin[{DirectoryName[$InputFileName], "ForScience-0.88.45.paclet"}]]];


BeginPackage["QuantumWalks`"]


(* For nice formatting of usage messages, see https://github.com/MMA-ForScience/ForScience *)
<<ForScience`;


ClearAll[
  Shift, Coin, DTQWStep, DTQW, PositionProbabilityDistribution
]


(* ::Section:: *)
(*Usage definitions*)


(* ::Subsection::Closed:: *)
(*DTQW*)


Shift::usage = FormatUsage["Shift[t] yields a sparse array of the Shift operator for a 1D DTQW in an infinite line at time ```t```."];
Coin::usage = FormatUsage["Coin[t] yields a sparse array of the Haddamard Coin operator for a 1D DTQW in an infinite line at time ```t```.
Coin[t,C] yields a sparse array of the Coin operator ```C``` for a 1D DTQW in an infinite line at time ```t```."];
DTQWStep::usage = FormatUsage["DTQWStep[t] yields the unitary matrix of a Haddamard 1D DTQW in an infinite line at time ```t```.
DTQWStep[t,C] yields the unitary matrix of a 1D DTQW in an infinite line, using coin ```C```, at time ```t```."];
DTQW::usage = FormatUsage["DTQW[\[Psi]_0,t] yields the state at time ```t``` of a 1D Haddamard DTQW in an infinite line with initial state ```\[Psi]_0```.
DTQW[\[Psi]_0,t,C] yields the state at time ```t``` of a 1D DTQW in an infinite line, with coin ```C```, with initial state ```\[Psi]_0```."];
PositionProbabilityDistribution::usage = FormatUsage["PositionProbabilityDistribution[\[Psi],t] yields the position probability distribution of the state ```\[Psi]``` of a 1D DTQW at time ```t```."];
ExpValPosition::usage = FormatUsage["ExpValPosition[\[Psi],t] returns the expected value of position for the state \[Psi] of a 1D DTQW at time ```t```."];


(* ::Subsection::Closed:: *)
(*Parrondo's paradox*)


L::usage = FormatUsage[
	"LoosingStrategy[\[Theta], \[Theta]_a, \[Theta]_b] returns the loosing inequality."
];


W::usage = FormatUsage[
	"WinningStrategy[\[Theta], \[Theta]_a, \[Theta]_b] returns the winning inequality."
];


CriticalAngle::usage = FormatUsage[
	"CriticalAngle[avgPos] takes a list of sublists ```avgPos```, where each subslit \
	is of the form '''{\[Theta],\[LeftAngleBracket]x(\[Theta])\[RightAngleBracket]}''', and returns the value '''\[Theta]''' such that \
	'''\[LeftAngleBracket]x(\[Theta])\[RightAngleBracket]''' is the closest to zero of all sublsists."
];


(* ::Section:: *)
(*Routine definitions*)


Begin["`Private`"]


(* ::Subsection::Closed:: *)
(*DTQW*)


Shift[t_] := Module[{},
  (* Check if t is an integer *)
  If[! IntegerQ[t], 
   Return[Message[Shift::intarg, t]]];
  
  (* Proceed with the original implementation *)
  SparseArray[
    Join[
      Table[{i, i + 2}, {i, Range[2, # - 2, 2]}], 
      Table[{i, i - 2}, {i, Range[3, #, 2]}]
    ] -> 1., {#, #}] &[2*(2*t + 1)]
]

(* Define a message for non-integer argument *)
Shift::intarg = "Argument `1` must be an integer (time).";


Coin::intarg = "Argument `1` must be an integer (time).";
Coin::matarg = "Argument `1` must be a matrix (coin argument).";

Coin[t_] := Module[{},
  (* Check if t is an integer *)
  If[! IntegerQ[t], 
   Return[Message[Coin::intarg, t]]];
  
  (* Proceed with the original implementation *)
  KroneckerProduct[IdentityMatrix[2 t + 1, SparseArray], SparseArray[FourierMatrix[2]]]
]

Coin[t_, c_] := Module[{},
  (* Check if t is an integer *)
  If[! IntegerQ[t], 
   Return[Message[Coin::intarg, t]]];
  
  (* Check if c is a matrix *)
  If[! MatrixQ[c], 
   Return[Message[Coin::matarg, c]]];
  
  (* Proceed with the original implementation *)
  KroneckerProduct[IdentityMatrix[2 t + 1, SparseArray], c]
]


DTQWStep[t_] := Shift[t] . Coin[t]
DTQWStep[t_, c_] := Shift[t] . Coin[t, c]
DTQWStep[t_, c_, psi_] := Chop[DTQWStep[t, c] . ArrayPad[psi, 2]]


DTQW[psi0_, t_] := Module[{psi},
  psi = ArrayPad[psi0, 2];
  Do[psi = ArrayPad[Chop[DTQWStep[i] . psi], 2], {i, t}];
  psi
]

DTQW[psi0_, t_, c_] := Module[{psi},
  psi = ArrayPad[psi0, 2];
  Do[psi = ArrayPad[Chop[DTQWStep[i, c] . psi], 2], {i, t}];
  psi
]


PositionProbabilityDistribution[psi_, tmax_] := Chop[
  Total[Abs[psi[[# ;; # + 1]]]^2] & /@ Range[1, 2*(2*tmax + 3), 2]
]


ExpValPosition[\[Psi]_,t_]:=PositionProbabilityDistribution[\[Psi],t] . Range[-t-1,t+1]


(* ::Subsection::Closed:: *)
(*Parrondo's paradox*)


L[\[Theta]_,\[Theta]a_,\[Theta]b_]:=\[Theta]a<=Mod[\[Theta],2.Pi]<=\[Theta]b


W[\[Theta]_,\[Theta]a_,\[Theta]b_]:=Mod[\[Theta],2.Pi]<=\[Theta]a||Mod[\[Theta],2.Pi]>=\[Theta]b


CriticalAngle[avgPos_] := 
	SortBy[
		Discard[avgPos, Round[#[[1]], 10.^-6] == Round[2Pi, 10.^-6] || Round[#[[1]], 10.^-6] == 0. &],
		Abs[#[[2]]]&
	][[1, 1]]


End[]


EndPackage[]




(* --- End: libsJA/QuantumWalks.wl --- *)



(* --- Begin: libsJA/QMB.wl --- *)


(* ::Package:: *)

(* If ForScience paclet not installed, install it. See https://github.com/MMA-ForScience/ForScience *)
If[Length[PacletFind["ForScience"]]==0, PacletInstall[FileNameJoin[{DirectoryName[$InputFileName], "ForScience-0.88.45.paclet"}]]];


(* ::Section:: *)
(*Begin package*)


BeginPackage["QMB`"];


(* For nice formatting of usage messages, see https://github.com/MMA-ForScience/ForScience *)
<<ForScience`;


(* ::Section:: *)
(*Notas*)


(* ::Text:: *)
(*Hay cosas en IAA_model.nb que 1) hay que migrar para ac\[AAcute] y 2) que hay que revisar si deber\[IAcute]a de poner ac\[AAcute]*)


(* ::Text:: *)
(*Hay cosas en los cuadernos del caometro donde hay rutinas para la secci\[OAcute]n de quantum chaos, como el unfolding etc*)


(* ::Text:: *)
(*Hay cosas de Heisenberg meets fuzzy que tambi\[EAcute]n tengo que pasar para ac\[AAcute]*)


(* ::Section:: *)
(*Usage definitions*)


(* ::Subsection::Closed:: *)
(*General quantum mechanics*)


(* All usage messages are evaluated quietly as FormatUsage[] requires FrontEnd. Therefore, if 
   QMB.wl is loaded in a .wls no error about FrontEnd pops up. *)


Quiet[
DensityMatrix::usage = FormatUsage[
"DensityMatrix[\[Psi]] returns the density matrix of state vector ```\[Psi]```."
];, {FrontEndObject::notavail, First::normal}];


Quiet[
Pauli::usage = FormatUsage[
"Pauli[0-3] gives the Pauli matrices. 
Pauli[{i_1,...,i_N}] returns the ```N```-qubit Pauli string '''Pauli[```i_1```]''' \[CircleTimes] '''...''' \[CircleTimes] '''Pauli[```i_N```]'''."];
, {FrontEndObject::notavail, First::normal}];


MatrixPartialTrace::usage = "MatrixPartialTrace[mat, n, d] calculates the partial trace of mat over the nth subspace, where all subspaces have dimension d.
MatrixPartialTrace[mat, n, {\!\(\*SubscriptBox[\(d\), \(1\)]\),\!\(\*SubscriptBox[\(d\), \(1\)]\),\[Ellipsis]}] calculates the partial trace of matrix mat over the nth subspace, where mat is assumed to lie in a space constructed as a tensor product of subspaces with dimensions {d1,d2,\[Ellipsis]}.";


VectorFromKetInComputationalBasis::usage = "VectorFromKetInComputationalBasis[ket] returns the matrix representation of ket.";


KetInComputationalBasisFromVector::usage = "KetInComputationalBasisFromVector[vector] returns the ket representation in computational basis of vector.";


Quiet[
RandomQubitState::usage = FormatUsage["RandomQubitState[] returns a Haar random qubit state."]
, {FrontEndObject::notavail, First::normal}];


Quiet[
RandomChainProductState::usage = FormatUsage[
"RandomChainProductState[L] returns a random ```L```-qubit product state."
], {FrontEndObject::notavail, First::normal}];


Quiet[
Dyad::usage = FormatUsage[
"Dyad[\[Psi]] returns ```|\[Psi]\[RightAngleBracket]\[LeftAngleBracket]\[Psi]|```.
Dyad[\[Psi],\[Phi]] returns ```|\[Psi]\[RightAngleBracket]\[LeftAngleBracket]\[Phi]|```."];
, {FrontEndObject::notavail, First::normal}];


Quiet[
Commutator::usage = FormatUsage["Commutator[A,B] returns AB - BA."];
, {FrontEndObject::notavail, First::normal}];


Quiet[
CommutationQ::usage = FormatUsage["CommutationQ[A,B] yields True if ```A``` and ```B``` commute, and False otherwise."];
, {FrontEndObject::notavail, First::normal}];


Quiet[
MutuallyCommutingSetQ::usage= FormatUsage[
"MutuallyCommutingSetQ[{A,B,...}] yields True if all matrices ```{A,B,...}``` mutually commute, and False otherwise."];
, {FrontEndObject::notavail, First::normal}];


Quiet[
Braket::usage = FormatUsage[
"Braket[\[Psi],\[Phi]] returns \[LeftAngleBracket]```\[Psi]```|```\[Phi]```\[RightAngleBracket]."
];
, {FrontEndObject::notavail, First::normal}];


FixCkForStateEvoultion::usage = "FixCkForStateEvoultion[\!\(\*SubscriptBox[\(\[Psi]\), \(0\)]\), { \!\(\*TemplateBox[{SubscriptBox[\"E\", \"k\"]},\n\"Ket\"]\) }] fixes \!\(\*SubscriptBox[\(c\), \(k\)]\) = \!\(\*TemplateBox[{RowBox[{SubscriptBox[\"E\", \"k\"], \" \"}], RowBox[{\" \", SubscriptBox[\"\[Psi]\", \"0\"]}]},\n\"BraKet\"]\) for StateEvolution[]";


StateEvolution::usage = "StateEvolution[t, \!\(\*SubscriptBox[\(\[Psi]\), \(0\)]\), {E_i}, {\!\(\*TemplateBox[{\"E_i\"},\n\"Ket\"]\)} ] returns \!\(\*TemplateBox[{RowBox[{\"\[Psi]\", RowBox[{\"(\", \"t\", \")\"}]}]},\n\"Ket\"]\) = \!\(\*SubscriptBox[\(\[Sum]\), \(\(\\ \)\(i\)\)]\) \!\(\*SuperscriptBox[\(\[ExponentialE]\), \(\(-\[ImaginaryI]\)\\  \*SubscriptBox[\(E\), \(i\)]\\  t\)]\)\!\(\*TemplateBox[{SubscriptBox[\"E\", \"i\"], RowBox[{\" \", SubscriptBox[\"\[Psi]\", \"0\"]}]},\n\"BraKet\"]\)\!\(\*TemplateBox[{SubscriptBox[\"E\", \"i\"]},\n\"Ket\"]\).
StateEvolution[t, {\!\(\*SubscriptBox[\(E\), \(k\)]\)}] calculates \!\(\*TemplateBox[{RowBox[{\"\[Psi]\", RowBox[{\"(\", \"t\", \")\"}]}]},\"Ket\"]\) = \!\(\*SubscriptBox[\(\[Sum]\), \(\(\\\\\)\(i\)\)]\) \!\(\*SuperscriptBox[\(\[ExponentialE]\), \(\(-\[ImaginaryI]\)\\\\\*SubscriptBox[\(E\), \(i\)]\\\\t\)]\)\!\(\*TemplateBox[{SubscriptBox[\"E\", \"i\"], RowBox[{\" \", SubscriptBox[\"\[Psi]\", \"0\"]}]},\"BraKet\"]\)\!\(\*TemplateBox[{SubscriptBox[\"E\", \"i\"]},\"Ket\"]\) having fixed the \!\(\*SubscriptBox[\(c\), \(k\)]\)'s with FixCkForStateEvoultion[\!\(\*SubscriptBox[\(\[Psi]\), \(0\)]\), { \!\(\*TemplateBox[{SubscriptBox[\"E\", \"k\"]},\n\"Ket\"]\) }].";


Quiet[
BlochVector::usage = FormatUsage["BlochVector[\[Rho]] returns the Bloch vector of a single-qubit density matrix \[Rho]."];
, {FrontEndObject::notavail, First::normal}];


KroneckerVectorProduct::usage = "KroneckerVectorProduct[a,b] calculates \!\(\*TemplateBox[{\"a\"},\n\"Ket\"]\)\[CircleTimes]\!\(\*TemplateBox[{\"b\"},\n\"Ket\"]\).";


Quiet[
Purity::usage = FormatUsage[
"Purity[\[Rho]] calculates the purity of ```\[Rho]```."
];
, {FrontEndObject::notavail, First::normal}];


Quiet[
Concurrence::usage = FormatUsage[
"Concurrence[\[Rho]] returns the two-qubit concurrence of density matrix ```\[Rho]```."
];
, {FrontEndObject::notavail, First::normal}];


Quiet[
Qubit::usage = FormatUsage[
"Qubit[\[Theta],\[Phi]] returns the state cos(```\[Theta]```/2)|0\[RightAngleBracket] + \[ExponentialE]^\[Phi] sin(```\[Theta]```/2)|1\[RightAngleBracket]"
];
, {FrontEndObject::notavail, First::normal}];


Quiet[
SU2Rotation::usage = FormatUsage[
	"SU2Rotation[{\[Theta]_a,\[Phi]_a},\[Theta]_R] the SU(2) matrix rotation with axis ```(\[Theta]_a,\[Phi]_a)``` \
	and angle rotation ```\[Theta]_R```.
	'''SU2Rotation[{```x,y,z```},```\[Theta]_R```]''' the SU(2) matrix rotation with axis ```(x,y,z)``` \
	and angle rotation ```\[Theta]_R```."
];
, {FrontEndObject::notavail, First::normal}];


(* ::Text:: *)
(*Agregadas por Miguel*)


coherentstate::usage = "coherentstate[state,L] Generates a spin coherent state of L spins given a general single qubit state";


(* ::Subsection::Closed:: *)
(*Quantum chaos*)


(*buscar la rutina del unfolding para meterla aqu\[IAcute]. Quiz\[AAcute]s tambi\[EAcute]n las cosas de wigner dyson y poisson*)


MeanLevelSpacingRatio::usage = FormatUsage[
	"MeanLevelSpacingRatio[eigenvalues] returns \[LeftAngleBracket]r_n\[RightAngleBracket]."
];


Quiet[
	IPR::usage = 
		FormatUsage["IPR[\[Psi]] computes the Inverse Participation Ratio of  ```\[Psi]``` in computational basis."];
, {FrontEndObject::notavail, First::normal}];


kthOrderSpacings::usage = FormatUsage[
	"kthOrderSpacings[spectrum,k] returns the ```k```-th order level spacing of ```spectrum```."
];


SpacingRatios::usage = FormatUsage[
	"SpacingRatios[spectrum,k] returns the ```k```-th order level spacing ratios of ```spectrum```."
];


(* ::Subsection::Closed:: *)
(*RMT*)


RatiosDistribution::usage = FormatUsage[
	"RatiosDistribution[r,\[Beta]] represents the probability distribution of level spacing \
	ratios P_\[Beta](r)."
];


RatiosDistributionPoisson::usage = FormatUsage[
	"RatiosDistributionPoisson[r,k] represents the probability distribution of level spacing \
	ratios P(r) of a Poissonian spectrum."
];


(* ::Subsection::Closed:: *)
(*Quantum channels*)


Reshuffle::usage = "Reshuffle[m] applies the reshuffle transformation to the matrix m with dimension \!\(\*SuperscriptBox[\(d\), \(2\)]\)\[Times]\!\(\*SuperscriptBox[\(d\), \(2\)]\).
Reshuffle[A,m,n] reshuffles matrix A, where dim(A) = mn.";


Quiet[
	SuperoperatorFromU::usage = FormatUsage["DensityMatrix[\[Psi]] returns the density matrix of state vector ```\[Psi]```."];
, {FrontEndObject::notavail, First::normal}];


(* ::Subsection::Closed:: *)
(*Bose-Hubbard*)


BoseHubbardHamiltonian::usage = FormatUsage[
  "BoseHubbardHamiltonian[n, L, J, U] returns the Bose-Hubbard Hamiltonian for \
```n``` bosons and ```L``` sites with hopping parameter ```J``` and interaction \
parameter ```U```.\n" <>
  "BoseHubbardHamiltonian[n, L, J, U, SymmetricSubspace] returns the Hamiltonian \
in a symmetric subspace. Option SymmetricSubspace takes the values \
\"All\" | \"EvenParity\" | \"OddParity\"."
];


SymmetricSubspace::usage = 
  "SymmetricSubspace is an option for BoseHubbardHamiltonian. Valid values are \
\"All\", \"EvenParity\", and \"OddParity\".";


KineticTermBoseHubbardHamiltonian::usage = FormatUsage[
"KineticTermBoseHubbardHamiltonian[basis] returns the kinetic term of the BH Hamiltonian \
with ```basis``` a list with Fock basis elements. \ 
KineticTermBoseHubbardHamiltonian[basis,SymmetricSubspace] returns the kinetic term of the BH \
Hamiltonian in a symmetric subspace with ```basis``` a list with Fock basis elements. \
Option SymmetricSubspace takes the values \"All\" | \"EvenParity\" | \"OddParity\"."
];


PotentialTermBoseHubbardHamiltonian::usage = FormatUsage[
  "PotentialTermBoseHubbardHamiltonian[n, L, SymmetricSubspace] returns the \
potential term of the Bose-Hubbard Hamiltonian for ```n``` bosons and ```L``` \
sites.
PotentialTermBoseHubbardHamiltonian[basis] returns the potential term of the \
Bose-Hubbard Hamiltonian given the ```basis``` of the Fock space of the bosonic \
system.\n\n"
  (*"**Notes**\n" <>
  "- If you want the potential term in a parity symmetry sector, ```basis``` \
should be a list containing only the representative Fock states of that \
symmetric subspace.\n\n" <>
  "**Examples of usage**\n" <>
  "- '''PotentialTermBoseHubbardHamiltonian[{{3,0,0},{2,1,0},{2,0,1},{1,2,0}}]''' \
returns the matrix in the odd parity sector of a system with 3 bosons and 3 \
sites."
*)];



BosonEscapeKrausOperators::usage = "BosonEscapeKrausOperators[N, L]: bosons escape to nearest neighbouring sites. N: bosons, L: site.";


BosonEscapeKrausOperators2::usage = "sdfa";


HilbertSpaceDim::replaced = "Function `1` has been replaced by `2`.";


BoseHubbardHilbertSpaceDimension::usage = FormatUsage["BoseHubbardHilbertSpaceDimension[n,L] returns the dimension"<> 
"of Hilbert space of a Bose Hubbard system of N bosons and L sites."];


FockBasis::usage = "FockBasis[N, M] returns the lexicographical-sorted Fock basis of N bosons in M sites.";


SortFockBasis::usage = "SortFockBasis[fockBasis] returns fockBasis in ascending-order according to the tag of Fock states.";


Tag::usage = "Tag[ { \!\(\*SubscriptBox[\(k\), \(1\)]\),\!\(\*SubscriptBox[\(k\), \(2\)]\),\[Ellipsis],\!\(\*SubscriptBox[\(k\), \(L\)]\) } ] returns the tag \!\(\*UnderoverscriptBox[\(\[Sum]\), \(i = 1\), \(L\)]\)\!\(\*SqrtBox[\(100  i + 3\)]\)\!\(\*SubscriptBox[\(k\), \(i\)]\) of Fock state \!\(\*TemplateBox[{RowBox[{SubscriptBox[\"k\", \"1\"], \",\", SubscriptBox[\"k\", \"2\"], \",\", \"\[Ellipsis]\", \",\", SubscriptBox[\"k\", \"L\"]}]},\n\"Ket\"]\).";


FockBasisStateAsColumnVector::usage = "FockBasisStateAsColumnVector[FockState, N, L] returns matrix representation of fockState={\!\(\*SubscriptBox[\(i\), \(1\)]\),\!\(\*SubscriptBox[\(i\), \(2\)]\),\[Ellipsis],\!\(\*SubscriptBox[\(i\), \(L\)]\)}. N: bosons, L: sites.";


FockBasisIndex::usage = "FockBasisIndex[fockState, sortedTagsFockBasis] returns the position of fockState in the tag-sorted Fock basis with tags sortedTagsFockBasis.";


RenyiEntropy::usage = "RenyiEntropy[\[Alpha], \[Rho]] computes the \[Alpha]-th order Renyi entropy of density matrix \[Rho].";


BosonicPartialTrace::usage = "BosonicPartialTrace[\[Rho]] calculates the partial trace of \[Rho]. Requires initialization.";


InitializationBosonicPartialTrace::usage = "InitializationBosonicPartialTrace[{\!\(\*SubscriptBox[\(i\), \(1\)]\),\[Ellipsis],\!\(\*SubscriptBox[\(i\), \(k\)]\)}, N, L] initializes variables for BosonicPartialTrace[] to calculate the reduced density matrix of sites {\!\(\*SubscriptBox[\(i\), \(1\)]\),\[Ellipsis],\!\(\*SubscriptBox[\(i\), \(k\)]\)}.";


(* ::Subsubsection::Closed:: *)
(*Fuzzy measurements in bosonic systems*)


InitializeVariables::usage = "InitializeVariables[n, L, boundaries, FMmodel] sets up the necessary variables for correct running of FuzzyMeasurement[\[Psi], \!\(\*SubscriptBox[\(p\), \(fuzzy\)]\)]; boundaries: 'open' or 'closed'; FMmodel: '#NN'.";


FuzzyMeasurement::usage = "FuzzyMeasurement[\[Psi], \!\(\*SubscriptBox[\(p\), \(fuzzy\)]\)] gives \[ScriptCapitalF](\!\(\*TemplateBox[{\"\[Psi]\"},\n\"Ket\"]\)\!\(\*TemplateBox[{\"\[Psi]\"},\n\"Bra\"]\)) = (1 - \!\(\*SubscriptBox[\(p\), \(fuzzy\)]\))\!\(\*TemplateBox[{\"\[Psi]\"},\n\"Ket\"]\)\!\(\*TemplateBox[{\"\[Psi]\"},\n\"Bra\"]\) + \!\(\*SubscriptBox[\(p\), \(fuzzy\)]\) \!\(\*UnderscriptBox[\(\[Sum]\), \(i\)]\) \!\(\*SubscriptBox[\(S\), \(i\)]\)\!\(\*TemplateBox[{\"\[Psi]\"},\n\"Ket\"]\)\!\(\*TemplateBox[{\"\[Psi]\"},\n\"Bra\"]\)\!\(\*SubsuperscriptBox[\(S\), \(i\), \(\[Dagger]\)]\), where \!\(\*SubscriptBox[\(S\), \(i\)]\) must be initizalized runnning InitializeVariables[n, L, boundaries, FMmodel].";


(* ::Subsection:: *)
(*Spin chains*)


(* ::Subsubsection::Closed:: *)
(*Symmetries*)


SpinParityEigenvectors::usage = "SpinParityEigenvectors[L] gives a list of {even, odd} eigenvectors of the L-spin system parity operator P; P\!\(\*TemplateBox[{RowBox[{SubscriptBox[\"k\", \"1\"], \",\", \"\[Ellipsis]\", \",\", SubscriptBox[\"k\", \"L\"]}]},\n\"Ket\"]\) = \!\(\*TemplateBox[{RowBox[{SubscriptBox[\"k\", \"L\"], \",\", \"\[Ellipsis]\", \",\", SubscriptBox[\"k\", \"1\"]}]},\n\"Ket\"]\), \!\(\*SubscriptBox[\(k\), \(i\)]\)=0,1.";


TranslationEigenvectorRepresentatives::usage = FormatUsage[
  "TranslationEigenvectorRepresentatives[L] returns a list of sublists, each containing:\
  the decimal representation of a bit-string representative eigenvector,\
  its pseudomomentum k, and the length of its translation orbit,\
  for a system of ```L``` qubits."
];


BlockDiagonalize::usage = FormatUsage[
	"BlockDiagonalize[matrix,opts] returns ```matrix``` in block-diagonal form. Only \
	option is '''Symmetry'''."
];


Symmetry::usage = FormatUsage[
	"Symmetry is an option for '''BlockDiagonalize''' to specify the symmetry according to which \
	```matrix``` in '''BlockDiagonalize'''[```matrix```] is block diagonalized. It takes the values: \
	\"Translation\". Default option is \"Translation\"."
];


(* ::Subsubsection::Closed:: *)
(*Hamiltonians*)


IsingHamiltonian::usage = FormatUsage[
	"IsingHamiltonian[h_x,h_z,J,L,opts] returns the Hamiltonian \ 
	H = \[Sum]_{*i=1*}^L (```h_x```\[Sigma]_i^x + ```h_z```\[Sigma]_i^z) - ```J``` \[Sum]_{*i=1*}^{*L-1*} \[Sigma]^z_i \[Sigma]^z_{*i+1*} \
	with boundary conditions specified by option BoundaryConditions (default is \"Open\")."
];


BoundaryConditions::usage = FormatUsage[
	"BoundaryConditions is an option for '''IsingHamiltonian''' to specify the boundary conditions. It \
	takes the values \"Open\" or \"Periodic\". Default option is \"Open\"."
];


IsingNNOpenHamiltonian::replaced = "Function `1` has been replaced by `2`.";


(*Quiet[
IsingNNOpenHamiltonian::usage = FormatUsage["IsingNNOpenHamiltonian[h_x,h_z,J,L] returns the Hamiltonian H = \[Sum]_{*i=1*}^L (```h_x```\[Sigma]_i^x + ```h_z```\[Sigma]_i^z) - ```J```\[Sum]_{*i=1*}^{*L-1*} \[Sigma]^z_i \[Sigma]^z_{*i+1*}.
IsingNNOpenHamiltonian[h_x,h_z,{J_1,...,J_L},L] returns the Hamiltonian H = \[Sum]_{*i=1*}^L (```h_x```\[Sigma]_i^x + ```h_z```\[Sigma]_i^z) - \[Sum]_{*i=1*}^{*L-1*} ```J_i``` \[Sigma]^z_i \[Sigma]^z_{*i+1*}."];
, {FrontEndObject::notavail, First::normal}];*)


IsingNNClosedHamiltonian::replaced = "Function `1` has been replaced by `2`.";


(*IsingNNClosedHamiltonian::usage = "IsingNNClosedHamiltonian[\!\(\*
StyleBox[SubscriptBox[\"h\", \"x\"],\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[SubscriptBox[\"h\", \"z\"],\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\"J\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\"L\",\nFontSlant->\"Italic\"]\)] returns the Hamiltonian \!\(\*UnderoverscriptBox[\(\[Sum]\), \(i = 1\), \(L\)]\)(\!\(\*SubscriptBox[\(h\), \(x\)]\) \!\(\*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(x\)]\) + \!\(\*SubscriptBox[\(h\), \(z\)]\) \!\(\*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(z\)]\)) + \!\(\*UnderoverscriptBox[\(\[Sum]\), \(i = 1\), L]\) \!\(\*SubscriptBox[\(J\), \(i\)]\) \!\(\*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(z\)]\)\!\(\*SubsuperscriptBox[\(\[Sigma]\), \(i + 1\), \(z\)]\) with \!\(\*SubscriptBox[\(\[Sigma]\), \(L + 1\)]\) = \!\(\*SubscriptBox[\(\[Sigma]\), \(1\)]\).";*)


ClosedXXZHamiltonian::usage = "ClosedXXZHamiltonian[\!\(\*
StyleBox[\"L\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\"\[CapitalDelta]\",\nFontSlant->\"Italic\"]\)] returns the closed XXZ 1/2-spin chain as in appendix A.1 of Quantum 8, 1510 (2024).";


OpenXXZHamiltonian::usage= "OpenXXZHamiltonian[\!\(\*
StyleBox[\"L\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\"\[CapitalDelta]\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\"h1\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\",\",\nFontSlant->\"Italic\"]\)\!\(\*
StyleBox[\"h2\",\nFontSlant->\"Italic\"]\)] returns the open XXZ 1/2-spin chain as in appendix A.2 of Quantum 8, 1510 (2024).";


Quiet[
LeaSpinChainHamiltonian::usage = FormatUsage["LeaSpinChainHamiltonian[J_{*xy*},J_z,\[Omega],\[Epsilon]_d,L,d] returns the spin-1/2 chain H = \[Sum]_{*i=1*}^{*L-1*} ```J_{*xy*}```(S^x_i S^x_{*i+1*} + S^y_i S^y_{*i+1*}) + ```J_z```S^z_i S^z_{*i+1*} + \[Sum]_{*i=1*}^{*L*} ```\[Omega]``` S^z_i + \[Epsilon]_d S^z_d. [Eq. (1) in Am. J. Phys. 80, 246\[Dash]251 (2012)]."];
, {FrontEndObject::notavail, First::normal}];


Quiet[
XXZOpenHamiltonian::usage = FormatUsage["XXZOpenHamiltonian[J_{*xy*},J_z,\[Omega],\[Epsilon]_d,L,d] returns the spin-1/2 chain \n H = \[Sum]_{*i=1*}^{*L-1*} ```J_{*xy*}```(S^x_i S^x_{*i+1*} + S^y_i S^y_{*i+1*}) + ```J_z```S^z_i S^z_{*i+1*} + \[Sum]_{*i=1*}^{*L*} ```\[Omega]``` S^z_i + \[Epsilon]_d S^z_d. \n [Eq. (1) in Am. J. Phys. 80, 246\[Dash]251 (2012)]."];
, {FrontEndObject::notavail, First::normal}];


HeisenbergXXXwNoise::usage="HeisenbergXXXwNoise[hz,L] returns the Heisenberg XXX spin 1/2 chain with noise: \!\(\*FormBox[\(H\\\  = \\\ \*FractionBox[\(1\), \(4\)]\\\ \(\*SubsuperscriptBox[\(\[Sum]\), \(i = 1\), \(L - 1\)]\\\ \((\*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(x\)]\\\ \*SubsuperscriptBox[\(\[Sigma]\), \(i + 1\), \(x\)]\\\  + \\\ \*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(y\)]\\\ \*SubsuperscriptBox[\(\[Sigma]\), \(i + 1\), \(y\)]\\\  + \\\ \*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(z\)]\\\ \*SubsuperscriptBox[\(\[Sigma]\), \(i + 1\), \(z\)])\)\)\\\  + \\\ \*FractionBox[\(1\), \(2\)]\\\ \(\*SubsuperscriptBox[\(\[Sum]\), \(i = 1\), \(L\)]\*SubsuperscriptBox[\(h\), \(i\), \(z\)]\\\ \*SubsuperscriptBox[\(\[Sigma]\), \(i\), \(z\)]\\\ \(\((open\\\ boundaries)\)\(.\)\)\)\),
TraditionalForm]\)";


(* ::Subsection::Closed:: *)
(*Fuzzy measurement channels*)


SwapMatrix::usage = "SwapMatrix[targetSite, wrongSite, N, L] returns the swap matrix that exchanges site targetSite with wrongSite for a system of N bosons and L sites.";


FuzzyMeasurementChannel::usage = 
"FuzzyMeasurementChannel[\[Rho], p, PermMatrices] returns (1-p) \[Rho] + \!\(\*FractionBox[\(p\), \(N - 1\)]\) \!\(\*SubscriptBox[\(\[Sum]\), \(i\)]\)\!\(\*SubscriptBox[\(S\), \(i, i + 1\)]\)\!\(\*SubscriptBox[\(\[Rho]S\), \(i, i + 1\)]\).
FuzzyMeasurementChannel[\[Rho], {\!\(\*SubscriptBox[\(p\), \(totalError\)]\), \!\(\*SubscriptBox[\(p\), \(NN\)]\), \!\(\*SubscriptBox[\(p\), \(SNN\)]\)}, {{\!\(\*SubscriptBox[\(S\), \(i, i + 1\)]\)}, {\!\(\*SubscriptBox[\(S\), \(i, i + 2\)]\)}}] returns \[ScriptCapitalE](\[Rho])=(1-\!\(\*SubscriptBox[\(p\), \(totalError\)]\))\[Rho] + \!\(\*SubscriptBox[\(p\), \(totalError\)]\)(\!\(\*FractionBox[SubscriptBox[\(p\), \(NN\)], \(L - 1\)]\)) \!\(\*SuperscriptBox[SubscriptBox[\(\[Sum]\), \(i = 1\)], \(L - 1\)]\) \!\(\*SubscriptBox[\(S\), \(i, i + 1\)]\) \[Rho] \!\(\*SubscriptBox[\(S\), \(i, i + 1\)]\) + \!\(\*SubscriptBox[\(p\), \(totalError\)]\)(\!\(\*FractionBox[SubscriptBox[\(p\), \(SNN\)], \(L - 2\)]\)) \!\(\*SuperscriptBox[SubscriptBox[\(\[Sum]\), \(i = 1\)], \(L - 2\)]\) \!\(\*SubscriptBox[\(S\), \(i, i + 2\)]\) \[Rho] \!\(\*SubscriptBox[\(S\), \(i, i + 2\)]\).";


(* ::Section:: *)
(*Beginning of Package*)


Begin["`Private`"];


(* ::Section:: *)
(*Routine definitions*)


(*no poner los nombres de funciones p\[UAcute]blicas porque se joden la definici\[OAcute]n de uso*)
ClearAll[SigmaPlusSigmaMinus, SigmaMinusSigmaPlus];


(* ::Subsection:: *)
(*General quantum mechanics*)


DensityMatrix[\[Psi]_] := Outer[Times, \[Psi], Conjugate[\[Psi]]]


Pauli[0]=Pauli[{0}]=SparseArray[{{1,0}, {0,1}}]; 
Pauli[1]=Pauli[{1}]=SparseArray[{{0,1}, {1,0}}]; 
Pauli[2]=Pauli[{2}]=SparseArray[{{0,-I},{I,0}}]; 
Pauli[3]=Pauli[{3}]=SparseArray[{{1,0}, {0,-1}}];
Pauli[Indices_List] := KroneckerProduct @@ (Pauli /@ Indices)


MatrixPartialTrace=ResourceFunction["MatrixPartialTrace"];


VectorFromKetInComputationalBasis[ket_]:=Normal[SparseArray[FromDigits[ket,2]+1->1,Power[2,Length[ket]]]]


KetInComputationalBasisFromVector[vector_]:=IntegerDigits[Position[vector,1][[1,1]]-1,2,Log[2,Length[vector]]]


RandomQubitState[] := 
Module[{x,y,z,\[Theta],\[Phi]},
	{x,y,z} = RandomPoint[Sphere[]];
	{\[Theta],\[Phi]}={ArcCos[z],Sign[y]ArcCos[x/Sqrt[x^2+y^2]]};
	{Cos[\[Theta]/2],Exp[I \[Phi]]Sin[\[Theta]/2]}
]


RandomChainProductState[0] := {1}
RandomChainProductState[1] := RandomQubitState[]
RandomChainProductState[L_] := Flatten[KroneckerProduct@@Table[RandomQubitState[],L]]


Dyad[a_]:=Outer[Times,a,Conjugate[a]]
Dyad[a_,b_]:=Outer[Times,a,Conjugate[b]]


Commutator[A_,B_]:=A . B-B . A


ZeroMatrix[d_]:=ConstantArray[0,{d,d}]


CommutationQ[A_,B_]:=Commutator[A,B]==ZeroMatrix[Length[A]]


MutuallyCommutingSetQ[ListOfMatrices_]:=Module[{SetLength=Length[ListOfMatrices]},
AllTrue[Table[CommutationQ@@ListOfMatrices[[{i,j}]],{i,SetLength-1},{j,i+1,SetLength}],TrueQ,2]
]


Braket[a_,b_] := Conjugate[a] . b


StateEvolution[t_,psi0_List,eigenvals_List,eigenvecs_List]:=
(*|\[Psi](t)\[RightAngleBracket] = Underscript[\[Sum], k] Subscript[c, k]\[ExponentialE]^(-Subscript[\[ImaginaryI]E, k]t)|Subscript[E, k]\[RightAngleBracket], Subscript[c, k]=\[LeftAngleBracket]Subscript[E, k]\[VerticalSeparator] Subscript[\[Psi], 0]\[RightAngleBracket]*)
	Module[{ck},
		ck = N[Chop[Conjugate[eigenvecs] . psi0]];
		N[Chop[Total[ ck * Exp[-I*eigenvals*N[t]] * eigenvecs]]]
	]


FixCkForStateEvoultion[\[Psi]0_, eigenvecs_] :=
	Module[{},
		ck = N[ Chop[ Conjugate[eigenvecs] . \[Psi]0 ] ];
		Heigenvecs = eigenvecs;
	]


StateEvolution[t_,eigenvals_List]:=
(*|\[Psi](t)\[RightAngleBracket] = Underscript[\[Sum], k] Subscript[c, k]\[ExponentialE]^(-Subscript[\[ImaginaryI]E, k]t)|Subscript[E, k]\[RightAngleBracket], Subscript[c, k]=\[LeftAngleBracket]Subscript[E, k]\[VerticalSeparator] Subscript[\[Psi], 0]\[RightAngleBracket]*)
	N[Chop[Total[ ck * Exp[-I*eigenvals*N[t]] * Heigenvecs]]]


BlochVector[\[Rho]_]:=Chop[Tr[Pauli[#] . \[Rho]]&/@Range[3]]


KroneckerVectorProduct[a_,b_]:=Flatten[KroneckerProduct[a,b]]


Purity[\[Rho]_]:=Tr[\[Rho] . \[Rho]] 


Concurrence[\[Rho]_] :=
Module[{\[Rho]tilde, R, \[Lambda]},
	\[Rho]tilde=# . Conjugate[\[Rho]] . #&[Pauli[{2,2}]];
	R=MatrixPower[# . \[Rho]tilde . #&[MatrixPower[\[Rho],1/2]],1/2];
	\[Lambda]=ReverseSort[Chop[Eigenvalues[R]]];
	Max[0,\[Lambda][[1]]-Total[\[Lambda][[2;;]]]]
]


Qubit[\[Theta]_,\[Phi]_] := {Cos[\[Theta]/2],Exp[I \[Phi]]Sin[\[Theta]/2]}


\[Alpha]


coherentstate[state_,L_]:=Flatten[KroneckerProduct@@Table[state,L]]


SU2Rotation[sphCoord_List?VectorQ,\[Theta]_]/; Length[sphCoord]==2 :=
	Module[{n={Sin[#1]Cos[#2], Sin[#1]Sin[#2], Cos[#1]} & @@ sphCoord},
		MatrixExp[-I * \[Theta]/2 * n . (Pauli /@ Range[3])]
	]

SU2Rotation[n_List?VectorQ,\[Theta]_]/; Length[n]==3 && Chop[Norm[n]-1]==0 :=
	MatrixExp[-I * \[Theta]/2 * n . (Pauli /@ Range[3])]


(* ::Subsection::Closed:: *)
(*Quantum chaos*)


MeanLevelSpacingRatio[eigenvalues_]:=Mean[Min/@Transpose[{#,1/#}]&[Ratios[Differences[Sort[eigenvalues]]]]]


IPR[\[Psi]_] := Total[\[Psi]^4]


kthOrderSpacings[spectrum_, k_] := RotateLeft[#, k] - # &[Sort[spectrum, Greater]][[;; -(k+1)]]


SpacingRatios[spectrum_, k_]:=RotateLeft[#, k]/# &[kthOrderSpacings[spectrum, k]][[;; -(k+1)]]


(* ::Subsection::Closed:: *)
(*RMT*)


RatiosDistribution[r_,\[Beta]_]:=1/Z[r,\[Beta]]*(r+r^2)^\[Beta]/(1+r+r^2)^(1+3 \[Beta]/2)

Z[r_,\[Beta]_]:=Integrate[(r+r^2)^\[Beta]/(1+r+r^2)^(1 + 3*\[Beta]/2),{r,0,Infinity}]


RatiosDistributionPoisson[r_, k_] := (2k -1)!*Power[r, k - 1]/(((k - 1)!)^2 * (1 + r)^(2*k))


(* ::Subsection::Closed:: *)
(*Quantum channels*)


(* ::Subsubsection::Closed:: *)
(*Reshuffle*)


Reshuffle[m_] := ArrayFlatten[ArrayFlatten/@Partition[Partition[ArrayReshape[#,{Sqrt[Dimensions[m][[1]]],Sqrt[Dimensions[m][[1]]]}]&/@m,Sqrt[Dimensions[m][[1]]]],Sqrt[Dimensions[m][[1]]]],1];


Reshuffle[A_,m_,n_] := ArrayFlatten[ArrayReshape[A, {m, n, m, n}]]


(* ::Subsection::Closed:: *)
(*Bosons*)


FockBasisStateAsColumnVector[FockBasisState_,N_,L_]:=Normal[SparseArray[Position[SortFockBasis[Normal[FockBasis[N,L]]][[2]],FockBasisState]->1,Binomial[N+L-1,L]]]


(* ------------------------------------------------------
FockBasis[N,M] returns the lexicographical-sorted Fock basis for N bosons in M sites.
New implementation as of 15/Jun/2025 uses more efficient algorithm based on IntegerPartitions.
------------------------------------------------------ *)
FockBasis[N_, M_] := ReverseSort[Catenate[Permutations[PadRight[#, M]] & /@ IntegerPartitions[N, M]]]

(* Old implementation preserved for reference *)
(*
FockBasis[N_, M_] := Module[{k, fockState},
    k = 1;
    Normal[Join[
        {fockState = SparseArray[{1 -> N}, {M}]},
        Table[
            fockState = SparseArray[Join[Table[i -> fockState[[i]], {i, k - 1}], {k -> fockState[[k]] - 1}], {M}];
            fockState[[k + 1]] = N - Total[fockState[[1 ;; k]]];
            k = Assignationk[M, N, fockState];
            fockState,
        HilbertSpaceDim[N, M] - 1]
    ]]
]
*)


SortFockBasis[fockBasis_]:=Transpose[Sort[{Tag[#],#}&/@fockBasis]]


(* ::Subsection::Closed:: *)
(*Bose-Hubbard*)


(* Mensajes para tipos incorrectos *)
BoseHubbardHamiltonian::int = 
    "n (`1`) and L (`2`) are expected to be integers.";
BoseHubbardHamiltonian::real = 
    "Hopping J (`1`) and interaction parameters U (`2`) are expected " <>
    "to be real numbers.";
BoseHubbardHamiltonian::badSymmetricSubspace = 
    "Opci\[OAcute]n SymmetricSubspace `1` inv\[AAcute]lida. " <>
    "Opciones v\[AAcute]lidas: \"All\", \"EvenParity\" o \"OddParity\".";

Options[BoseHubbardHamiltonian] = {
    SymmetricSubspace -> "All" (* "All"|"EvenParity"|"OddParity" *),
    Version -> "New"
};

OptionValuePatterns[BoseHubbardHamiltonian] = {
  SymmetricSubspace -> Alternatives["All", "EvenParity", "OddParity"],
  Version -> _
};

BoseHubbardHamiltonian[n_Integer, L_Integer, J_Real, U_Real, 
    OptionsPattern[]] := 
Module[
    {tags, basis, basiseven, rbasiseven, rbasisodd, basisodd, H, T, V, map},
    
    basis = N[FockBasis[n, L]];
    H = -J*KineticTermBoseHubbardHamiltonian[basis] + 
        U/2*PotentialTermBoseHubbardHamiltonian[basis];
    
    (* Para subespacios de simetria, obtenerlos a partir de H *)
    Switch[OptionValue[SymmetricSubspace],
        "All",
            Nothing,
            
        "EvenParity",
            basiseven = DeleteDuplicatesBy[basis, Sort[{#, Reverse[#]}]&];
            rbasiseven = Reverse /@ basiseven;
            map = AssociationThread[
                basis -> Range[BoseHubbardHilbertSpaceDimension[n, L]]];
            H = 1/2 # . (H[[#1,#1]] + H[[#1,#2]] + H[[#2,#1]] + H[[#2,#2]] & @@ 
                Map[map, {basiseven, rbasiseven}, {2}]) . # &[
                DiagonalMatrix[
                    ReplacePart[
                        ConstantArray[1., Length[basiseven]],
                        Thread[
                            Flatten[Position[
                                basiseven, 
                                _?(PalindromeQ[#] &), 
                                {1}
                            ]] -> 1/Sqrt[2.]
                        ]
                    ],
                    TargetStructure -> "Sparse"
                ]
            ],
            
        "OddParity",
            basisodd = DeleteDuplicatesBy[
                Discard[basis, PalindromeQ], 
                Sort[{#, Reverse[#]}]&];
            rbasisodd = Reverse /@ basisodd;
            map = AssociationThread[
                basis -> Range[BoseHubbardHilbertSpaceDimension[n, L]]];
            H = 1/2 (H[[#1,#1]] - H[[#1,#2]] - H[[#2,#1]] + H[[#2,#2]] & @@ 
                Map[map, {basisodd, rbasisodd}, {2}]),
                
        _,
            Message[
                BoseHubbardHamiltonian::badSymmetricSubspace, 
                OptionValue[SymmetricSubspace]
            ];
            Return[$Failed];
    ];
    
    H
]

(* Handle cases where arguments don't match the expected types *)
BoseHubbardHamiltonian[N_, L_, J_, U_, OptionsPattern[]] := Module[{},
  If[!IntegerQ[N] || !IntegerQ[L],
    Message[BoseHubbardHamiltonian::int, N, L];
    Return[$Failed];
  ];
  If[Head[J] =!= Real || Head[U] =!= Real,
    Message[BoseHubbardHamiltonian::real, J, U];
    Return[$Failed];
  ];
];

SyntaxInformation[BoseHubbardHamiltonian] = <|
  "ArgumentsPattern" -> {_, _, _, _, OptionsPattern[]}
|>;


ClearAll[PotentialTermBoseHubbardHamiltonian];

(* Mensajes de error *)
PotentialTermBoseHubbardHamiltonian::nint = 
    "El n\[UAcute]mero de part\[IAcute]culas n (`1`) debe ser un entero positivo.";
PotentialTermBoseHubbardHamiltonian::lint = 
    "El n\[UAcute]mero de sitios L (`1`) debe ser un entero positivo.";
PotentialTermBoseHubbardHamiltonian::int = 
    "Los par\[AAcute]metros n (`1`) y L (`2`) deben ser enteros positivos.";
PotentialTermBoseHubbardHamiltonian::empty = 
    "La base proporcionada est\[AAcute] vac\[IAcute]a.";
PotentialTermBoseHubbardHamiltonian::dim = 
    "Todos los estados en la base deben tener la misma longitud.";
    
PotentialTermBoseHubbardHamiltonian[n_Integer?Positive, L_Integer?Positive] := 
DiagonalMatrix[
    Total /@ ((#^2 - #) &[FockBasis[n, L]]),
    TargetStructure -> "Sparse"
]

(* Versi\[OAcute]n con validaci\[OAcute]n de basis *)
PotentialTermBoseHubbardHamiltonian[basis_List] := 
Module[
    {lengths},
    If[basis === {},
        Message[PotentialTermBoseHubbardHamiltonian::empty];
        Return[$Failed]
    ];
    
    lengths = Length /@ basis;
    If[!AllTrue[lengths, EqualTo[First[lengths]]],
        Message[PotentialTermBoseHubbardHamiltonian::dim];
        Return[$Failed]
    ];
    
    DiagonalMatrix[
        Total /@ ((#^2 - #) &[basis]),
        TargetStructure -> "Sparse"
    ]
]

(* Versi\[OAcute]n con validaci\[OAcute]n de par\[AAcute]metros *)
PotentialTermBoseHubbardHamiltonian[n_Integer, L_Integer] := 
If[n <= 0 || L <= 0,
    Message[PotentialTermBoseHubbardHamiltonian::int, n, L];
    Return[$Failed]
];

PotentialTermBoseHubbardHamiltonian[n_, L_] /; !IntegerQ[n] := 
(
    Message[PotentialTermBoseHubbardHamiltonian::nint, n];
    $Failed
)

PotentialTermBoseHubbardHamiltonian[n_, L_] /; !IntegerQ[L] := 
(
    Message[PotentialTermBoseHubbardHamiltonian::lint, L];
    $Failed
)

PotentialTermBoseHubbardHamiltonian[___] := 
(
    Message[PotentialTermBoseHubbardHamiltonian::usage];
    $Failed
)


KineticTermBoseHubbardHamiltonian::badSymmetricSubspace = 
    "Opci\[OAcute]n SymmetricSubspace `1` inv\[AAcute]lida. " <>
    "Opciones v\[AAcute]lidas: \"All\", \"EvenParity\" o \"OddParity\".";

Options[KineticTermBoseHubbardHamiltonian] = {
    SymmetricSubspace -> "All" (* "All"|"EvenParity"|"OddParity" *)
};

KineticTermBoseHubbardHamiltonian[basis_, OptionsPattern[]] := 
Module[
    {len = Length[basis], basisNumRange, T, basiseven, rbasiseven, 
     basisodd, rbasisodd, map},
    
    basisNumRange = Range[len];
    T = # + ConjugateTranspose[#] & [
        SparseArray[
            AssignationRulesKinetic[basis, 
             AssociationThread[basis -> basisNumRange], basisNumRange], 
            {len, len}
        ]
    ];
    
    Switch[OptionValue[SymmetricSubspace],
        "All",
            Nothing,
        "EvenParity",
            basiseven = DeleteDuplicatesBy[basis, Sort[{#, Reverse[#]}]&];
            rbasiseven = Reverse /@ basiseven;
            map = AssociationThread[basis -> Range[len]];
            T = 1/2 # . (T[[#1,#1]] + T[[#1,#2]] + T[[#2,#1]] + T[[#2,#2]] & @@ 
                Map[map, {basiseven, rbasiseven}, {2}]) . # &[
                DiagonalMatrix[
                    ReplacePart[
                        ConstantArray[1., Length[basiseven]],
                        Thread[
                            Flatten[Position[
                                basiseven, 
                                _?(PalindromeQ[#] &), 
                                {1}
                            ]] -> 1/Sqrt[2.]
                        ]
                    ],
                    TargetStructure -> "Sparse"
                ]
            ],
        "OddParity",
            basisodd = DeleteDuplicatesBy[
                Discard[basis, PalindromeQ], Sort[{#, Reverse[#]}]&];
            rbasisodd = Reverse /@ basisodd;
            map = AssociationThread[basis -> Range[len]];
            T = 1/2 (T[[#1,#1]] - T[[#1,#2]] - T[[#2,#1]] + T[[#2,#2]] & @@ 
                Map[map, {basisodd, rbasisodd}, {2}]),
            
        _,
            Message[
                KineticTermBoseHubbardHamiltonian::badSymmetricSubspace,
                OptionValue[SymmetricSubspace]
            ];
            Return[$Failed];
    ];
    
    T
]


AssignationRulesKinetic[basis_, positionMap_, basisNumRange_] := 
Catenate[
    MapThread[
        AssignationRulesKineticMapFunc,
        {
            Apply[
                {positionMap[#1], #2} &,
                DeleteCases[
                    Transpose[
                        {
                            StateAfterADaggerA[basis],
                            ValueAfterADaggerA[basis]
                        },
                        {3, 1, 2}
                    ],
                    {_, 0.},
                    {2}
                ],
                {2}
            ],
            basisNumRange
        }
    ]
]


StateAfterADaggerA[basis_] := 
Module[{len = Length[First[basis]]},
    Outer[
        Plus,
        basis,
        #,
        1
    ] & [
        Catenate[
            NestList[
                RotateRight,
                PadRight[#, len],
                len - 2
            ] & /@ {{1, -1}}
        ]
    ]
]


ValueAfterADaggerA[basis_] := 
MapApply[
    Sqrt[(#1 + 1.) * #2] &,
    Partition[#, 2, 1]
] & /@ basis


AssignationRulesKineticMapFunc[stateValuePairs_,index_] := 
({index, #1} -> #2) & @@@ stateValuePairs


Tag[FockBasisElement_]:=N[Round[Sum[Sqrt[100 i+3]#[[i+1]],{i,0,Length[#]-1}]&[FockBasisElement],10^-8]]
Tag[Nothing]:=Nothing


InitializationBosonicPartialTrace[SitesOfSubsystem_, n_, L_] :=
Module[{SubsystemSize,SubsystemComplementSize,SystemFockBasisTags,SystemFockBasis,SubsystemFockBasisTags,SubsystemFockBasis,SubsystemComplementFockBasis,RulesForOrderedSystemBasis,RulesForOrderedSubsystemBasis,FockIndicesInRho,FockIndicesInReducedRho},
SubsystemSize=Length[SitesOfSubsystem];
SubsystemComplementSize=L-SubsystemSize;
(*System's Fock basis*)
{SystemFockBasisTags,SystemFockBasis}=SortFockBasis[FockBasis[n,L]];
(*Subsystem's Fock basis*)
{SubsystemFockBasisTags,SubsystemFockBasis}=SortFockBasis[Flatten[Table[FockBasis[k,SubsystemSize],{k,0,n}],1]];
(*Complement subsystem's Fock basis*)
SubsystemComplementFockBasis=Map[ReplacePart[ConstantArray[_,L],Thread[Complement[Range[L],SitesOfSubsystem]->#]]&,Flatten[Table[SortFockBasis[FockBasis[k,SubsystemComplementSize]][[2]],{k,0,n}],1]];(*<<<*)
RulesForOrderedSystemBasis=Thread[Rule[SystemFockBasisTags,Range[BoseHubbardHilbertSpaceDimension[n,L]]]];
SubsystemHilbertSpaceDim=Length[SubsystemFockBasis];
RulesForOrderedSubsystemBasis=Thread[Rule[SubsystemFockBasisTags,Range[SubsystemHilbertSpaceDim]]];
FockIndicesInRho=Map[Tuples[{#,#}]&[Extract[SystemFockBasis,Position[SystemFockBasis,#]]]&,SubsystemComplementFockBasis];(*<<<*)
FockIndicesInReducedRho=Map[#[[All,All,SitesOfSubsystem]]&,FockIndicesInRho];(*<<<*)
ComputationalIndicesInRho=ReplaceAll[Map[Tag,FockIndicesInRho,{3}],RulesForOrderedSystemBasis];(*<<<*)
ComputationalIndicesInReducedRho=ReplaceAll[Map[Tag,FockIndicesInReducedRho,{3}],RulesForOrderedSubsystemBasis];(*<<<*)];


BosonicPartialTrace[Rho_] := 
	Module[{MatrixElementsOfRho,rules},
	
		MatrixElementsOfRho=Extract[Rho,#]&/@ComputationalIndicesInRho;
		rules=MapThread[Thread[Rule[#1,#2]]&,{ComputationalIndicesInReducedRho,MatrixElementsOfRho}];
		Total[Map[SparseArray[#,{SubsystemHilbertSpaceDim,SubsystemHilbertSpaceDim}]&,rules]]
	]


(* ::Subsubsection::Closed:: *)
(*Fuzzy measurements in bosonic systems*)


(* Initialization function *)
InitializeVariables[n_, L_, boundaries_, FMmodel_] := 
 Module[{basis, SwapAppliedToBasis},
  
  indices = Which[
    boundaries == "open",
    Table[ReplacePart[Range[L], {i -> Mod[i + 1, L, 1], Mod[i + 1, L, 1] -> i}], {i, L - ToExpression[StringTake[FMmodel, 1]]}],
    boundaries == "closed" || boundaries == "close",
    Print["Yet to come"]
  ];
  
  permutedBasisIndices = Table[
    basis = SortFockBasis[FockBasis[n, L]][[2]];
    SwapAppliedToBasis = #[[i]] & /@ basis;
    Flatten[Position[basis, #] & /@ SwapAppliedToBasis],
    {i, indices}
  ];
  
  numberOfPermutations = Length[permutedBasisIndices];
];


FuzzyMeasurement[\[Psi]_,pFuzzy_] := 
(1 - pFuzzy) Dyad[\[Psi]] + (pFuzzy/numberOfPermutations) * Total[ Table[ Dyad[\[Psi][[i]]], {i,permutedBasisIndices}]]


(* ::Subsection::Closed:: *)
(*BosonEscapeKrausOperators[n, L]*)


BosonEscapeKrausOperators[n_,L_] := Module[{dimH=BoseHubbardHilbertSpaceDimension[n,L],fockBasisTags,fockBasisStates,KrausOperators},
{fockBasisTags,fockBasisStates}=SortFockBasis[FockBasis[n,L]];

KrausOperators=1/Sqrt[2(n-1)]*Join[
Table[Normal[SparseArray[Thread[
Select[Table[Module[{fockState=fockBasisStates[[k]]},
{FockBasisIndex[SigmaPlusSigmaMinus[fockState,i,L],fockBasisTags],k}],
{k,dimH}],AllTrue[#,Positive]&]->1.],{dimH,dimH}]],
{i,n-1}],
Table[Normal[SparseArray[Thread[
Select[Table[Module[{fockState=fockBasisStates[[k]]},
{FockBasisIndex[SigmaMinusSigmaPlus[fockState,i,L],fockBasisTags],k}],
{k,dimH}],AllTrue[#,Positive]&]->1.],{dimH,dimH}]],
{i,n-1}]];
Append[KrausOperators,Sqrt[IdentityMatrix[dimH]-Sum[ConjugateTranspose[K] . K,{K,KrausOperators}]]]
]


SigmaPlusSigmaMinus[fockState_,i_,L_] := 
If[fockState[[i]]==L||fockState[[i+1]]==0,
Nothing,
ReplaceAt[ReplaceAt[fockState,x_:>x+1,i],x_:>x-1,i+1]
]


SigmaMinusSigmaPlus[fockState_,i_,L_] := 
If[fockState[[i]]==0||fockState[[i+1]]==L,
Nothing,
ReplaceAt[ReplaceAt[fockState,x_:>x-1,i],x_:>x+1,i+1]
]


(*Secondary routines*)


HilbertSpaceDim[args___] := Message[HilbertSpaceDim::replaced, "HilbertSpaceDim", "BoseHubbardHilbertSpaceDimension"];


BoseHubbardHilbertSpaceDimension[n_,L_]:=(n+L-1)!/(n!(L-1)!)


FockBasisIndex[fockState_,sortedTagsFockBasis_]:=
FromDigits[Flatten[Position[sortedTagsFockBasis,Tag[fockState]],1]]


(*Check definitions*)


Assignationk[M_,N_,n_]:=If[n[[1;;M-1]]==ConstantArray[0,M-1],M-1,FromDigits[Last[Position[Normal[n[[1;;M-1]]],x_ /;x!=0]]]]


RenyiEntropy[\[Alpha]_,\[Rho]_]:=1/(1-\[Alpha]) Log[Tr[MatrixPower[\[Rho],\[Alpha]]]]


(* ::Subsection::Closed:: *)
(*Spins*)


(* ::Subsubsection::Closed:: *)
(*Symmetries*)


SpinParityEigenvectors[L_]:=Module[{tuples,nonPalindromes,palindromes},
tuples=Tuples[{0,1},L];
nonPalindromes=Select[tuples,#!=Reverse[#]&];
palindromes=Complement[tuples,nonPalindromes];
nonPalindromes=DeleteDuplicatesBy[nonPalindromes,Sort[{#,Reverse[#]}]&];
Normal[
{
Join[SparseArray[FromDigits[#,2]+1->1.,2^L]&/@palindromes,Normalize[SparseArray[{FromDigits[#,2]+1->1.,FromDigits[Reverse[#],2]+1->1.},2^L]]&/@nonPalindromes],
Normalize[SparseArray[{FromDigits[#,2]+1->-1.,FromDigits[Reverse[#],2]+1->1.},2^L]]&/@nonPalindromes
}]
]


Translation[state_, L_] := BitShiftRight[state] + BitAnd[state, 1]*2^(L-1)
BinaryNecklaces[L_Integer] := Module[{tuples=Tuples[{0,1},L]},
	Union[Table[First[Sort[NestList[RotateLeft, t, L-1]]], {t,tuples}]]
]
\[Omega][L_,k_] := Exp[2Pi*k*I/L]

TranslationEigenvectorRepresentatives[L_Integer] := Module[
	{
	necklaces = FromDigits[#,2]& /@ BinaryNecklaces[L],
	orbits
	},
	
	orbits = DeleteDuplicates[
	Sort[NestWhileList[Translation[#, L]&, #, UnsameQ[##]&, All][[;;-2]]]& /@ necklaces
	 ];
	Catenate[
		Outer[
			If[Mod[Length[#1]*#2, L]==0, {First[#1], #2, Length[#1]}, Nothing]&, 
			orbits, 
			Range[0, L-1], 
			1
		]
	]
]


RepresentativesOddBasis[basis_]:=DeleteDuplicatesBy[Discard[basis,PalindromeQ],Sort[{#,Reverse[#]}]&]
RepresentativesEvenBasis[basis_]:=DeleteDuplicatesBy[basis,Sort[{#,Reverse[#]}]&]


Options[BlockDiagonalize] = {
  Symmetry -> "Translation" (*coming soon: \"Parity\"*)
};

BlockDiagonalize[A_, opts:OptionsPattern[]]:= 
Switch[OptionValue[Symmetry],
	"Translation",
		Module[
			{
				L = Log[2, Length[A]],
				repsgathered,
				P
			},
				(*Gather translation eigenvectors by their pseudomomentum k*)
				repsgathered = GatherBy[TranslationEigenvectorRepresentatives[L], #[[2]]&];
				
				(*change of basis matrix*)
				P = SparseArray[
						Catenate[
							Apply[
								Normalize[SparseArray[
									Thread[
										FoldList[Translation[#, L]&, #1, Range[#3 - 1]] + 1 -> 
										Power[\[Omega][L, #2], Range[0., #3 - 1]]
									]
										, 2^L
								]]&,
								repsgathered,
								{2}
							]
						]
					];
					
				BlockDiagonalMatrix[Chop[Conjugate[P] . A . Transpose[P]]]
		],
	"Parity",
		Module[
		{
			L = Log[2, Length[A]],
			basis, reps, rules, Heven, Hodd
		},
			basis = Tuples[{0, 1}, L];
			rules=AssociationThread[basis->Range[Length[basis]]];
			reps=Comap[{RepresentativesEvenBasis, RepresentativesOddBasis},basis];
			Heven=1/2 (A[[#1,#1]] + A[[#1,#2]] + A[[#2,#1]] + A[[#2,#2]] & @@ Map[rules, {#, Reverse/@#}&[reps[[1]]], {2}]);
			Hodd=1/2 (A[[#1,#1]] - A[[#1,#2]] - A[[#2,#1]] + A[[#2,#2]] & @@ Map[rules, {#, Reverse/@#}&[reps[[2]]], {2}]);
			Heven = # . Heven . #&[DiagonalMatrix[ReplacePart[ConstantArray[1.,Length[reps[[1]]]],Thread[Catenate[Position[reps[[1]],_?PalindromeQ,1]]->1/Sqrt[2.]]],TargetStructure->"Sparse"]];
			
			{Heven, Hodd}
		],
	_,
		Message[BlockDiagonalize::badSymmetry, OptionValue[Symmetry]];
		Return[$Failed];
]

(*Mensaje de error si la opci\[OAcute]n es inv\[AAcute]lida*)
BlockDiagonalize::badSymmetry = 
  "Option badSymmetry -> `1` is not valid. Valid options: \"Translation\".";


(* ::Subsubsection::Closed:: *)
(*Hamiltonians*)


Options[IsingHamiltonian] = {
  BoundaryConditions -> "Open"
};

IsingHamiltonian[hx_, hz_, J_, L_, opts:OptionsPattern[]] := Module[
	{NNIndices},
	NNIndices = Switch[OptionValue[BoundaryConditions],
		"Open",
			Normal[SparseArray[Thread[{#,#+1}->3],{L}]&/@Range[L-1]],
		"Periodic",
			Normal[SparseArray[Thread[{#,Mod[#+1,L,1]}->3],{L}]&/@Range[L]],
		_,
			Message[
                IsingHamiltonian::badBoundaryCondition, 
                OptionValue[BoundaryConditions]
            ];
            Return[$Failed];
	];
	
	Total[{hx*Pauli[#]+hz*Pauli[3#]&/@IdentityMatrix[L],-J*(Pauli/@NNIndices)},2]
]

(*Mensaje de error si la opci\[OAcute]n es inv\[AAcute]lida*)
IsingHamiltonian::badBoundaryCondition = 
  "Option BoundaryConditions -> `1` not valid. Valid options: \"Open\" o \"Periodic\".";


IsingNNOpenHamiltonian[args___] := Message[
	IsingNNOpenHamiltonian::replaced, "IsingNNOpenHamiltonian", "IsingHamiltonian"];
	
(*IsingNNOpenHamiltonian[hx_,hz_,J_,L_] := Module[{NNIndices},
	NNIndices=Normal[SparseArray[Thread[{#,#+1}->3],{L}]&/@Range[L-1]];
	N[Normal[Total[{hx*Pauli[#]+hz*Pauli[3#]&/@IdentityMatrix[L],-J*(Pauli/@NNIndices)},2]]]]*)


IsingNNClosedHamiltonian[args___] := Message[
	IsingNNClosedHamiltonian::replaced, "IsingNNClosedHamiltonian", "IsingHamiltonian"];

(*IsingNNClosedHamiltonian[hx_,hz_,J_,L_] := 
Module[{NNIndices},
	NNIndices=Normal[SparseArray[Thread[{#,Mod[#+1,L,1]}->3],{L}]&/@Range[L]];
	Total[{hx*Pauli[#]+hz*Pauli[3#]&/@IdentityMatrix[L],-J*(Pauli/@NNIndices)},2]
]*)


ClosedXXZHamiltonian[L_,\[CapitalDelta]_]:=
	Module[{NNindices},
		NNindices = Normal[ SparseArray[Thread[{#, Mod[# + 1, L, 1]}->1], {L}] &/@ Range[L] ];
		N[Normal[-1/2*Total[Join[Pauli/@NNindices,Pauli/@(2NNindices),\[CapitalDelta] (Pauli[#]-IdentityMatrix[2^L])&/@(3NNindices)]]]]
	]


OpenXXZHamiltonian[L_,\[CapitalDelta]_,h1_,h2_]:=
	Module[{NNindices},
		NNindices = Normal[ SparseArray[Thread[{#, Mod[# + 1, L, 1]}->1], {L}] &/@ Range[L-1] ];
		N[Normal[-1/2*Total[Join[Pauli/@NNindices,Pauli/@(2NNindices),\[CapitalDelta] (Pauli[#]-IdentityMatrix[2^L])&/@(3NNindices)]]  
		- 1/2*(h1 Pauli[Join[{1},ConstantArray[0,L-1]]] + h2*Pauli[Join[ConstantArray[0,L-1],{1}]])+ 1/2*(h1 + h2)IdentityMatrix[2^L]]]
	]


HamiltonianNN[Jxy_,Jz_,L_]:=
	Module[{NNindices},
		NNindices = Normal[ SparseArray[Thread[{#, Mod[# + 1, L, 1]}->1], {L}] &/@ Range[L-1] ];
		N[Normal[(1/4)*Total[Join[Jxy*(Pauli/@NNindices),Jxy*(Pauli/@(2NNindices)),Jz*(Pauli[#]&/@(3NNindices))]]]]
	]

HamiltonianZ[\[Omega]_,\[Epsilon]d_,L_,d_]:=N[(1/2)*(\[Omega]*Total[Pauli/@(3*IdentityMatrix[L])]+\[Epsilon]d*Pauli[Normal[SparseArray[d->3,L]]])]

LeaSpinChainHamiltonian[Jxy_,Jz_,\[Omega]_,\[Epsilon]d_,L_,d_]:=HamiltonianNN[Jxy,Jz,L]+HamiltonianZ[\[Omega],\[Epsilon]d,L,d]
XXZOpenHamiltonian[Jxy_,Jz_,\[Omega]_,\[Epsilon]d_,L_,d_]:=HamiltonianNN[Jxy,Jz,L]+HamiltonianZ[\[Omega],\[Epsilon]d,L,d]


HeisenbergXXXwNoise[h_List,L_]:=
Module[{NNIndices,firstSum,secondSum},
(* \sum_{k=1}^{L-1} S_k^xS_{k+1}^x + S_k^zS_{k+1}^z + S_k^zS_{k+1}^z *)
NNIndices=Normal[SparseArray[Thread[{#,#+1}->1],{L}]&/@Range[L-1]];
firstSum=1/4*Total[Table[Pauli[i*#]&/@NNIndices,{i,3}],2];

(* \sum_{k=1}^{L} h_k^z S_k^z *)
secondSum=1/2*h . (Pauli/@DiagonalMatrix[ConstantArray[3,L]]);

firstSum+secondSum
]


(* ::Subsection::Closed:: *)
(*Fuzzy measurement channels*)


SwapMatrix[targetSite_, wrongSite_, N_, L_] := Module[
    {
        tagsOfSwappedFockBasis, (* List to hold tags of swapped Fock basis *)
        sortedTags,             (* List to hold sorted tags *)
        sortedFockBasis         (* List to hold sorted Fock basis *)
    },
    
    (* Step 1: Generate and sort the Fock basis (according to its tags) *)
    {sortedTags, sortedFockBasis} = Normal[SortFockBasis[FockBasis[N, L]]];
    
    (* Step 2: Generate tags for the swapped Fock basis *)
    tagsOfSwappedFockBasis = Tag /@ Map[SwappedFockState[#, targetSite, wrongSite] &, sortedFockBasis];
    
    (* Step 3: Create the sparse matrix corresponding to the swap matrix *)
    SparseArray[
        Table[{i, Position[sortedTags, tagsOfSwappedFockBasis[[i]]][[1, 1]]},
        {i, Length[tagsOfSwappedFockBasis]}] -> 1]
]


FuzzyMeasurementChannel[\[Rho]_, pTotalError_, SwapNNmatrices_]:=(1 - pTotalError) \[Rho] +
	pTotalError 1/Length[SwapNNmatrices] Total[# . \[Rho] . # & /@ SwapNNmatrices]


FuzzyMeasurementChannel[\[Rho]_, pErrors_List, SwapMatrices_] := Module[
	{
		{pTotalError, pNN, pSNN} = pErrors,
		{SwapNN, SwapSNN} = SwapMatrices
	},
	
	(1 - pTotalError) \[Rho] + pNN ( pTotalError/Length[SwapNN] ) Total[# . \[Rho] . # & /@ SwapNN]
	+ pNN ( pTotalError/Length[SwapSNN] ) Total[# . \[Rho] . # & /@ SwapSNN]
]


(* ::Subsubsection:: *)
(*Private functions*)


SwappedFockState[fockState_, targetSite_, wrongSite_] := 
  ReplacePart[fockState, 
    Thread[# -> Part[fockState, Reverse[#]] ] & [{targetSite, wrongSite}]
  ]


(* ::Section:: *)
(*End of Package*)


End[];


EndPackage[];




(* --- End: libsJA/QMB.wl --- *)



(* --- Begin: QW/PacletInfo.wl --- *)


(* ::Package:: *)

PacletObject[
    <|
        "Name" -> "QW",
        "Version" -> "0.1.0",
        "WolframVersion" -> "13.3+",
        "Extensions" ->
            {
                {
                    "Kernel",
                    "Root" -> "Kernel",
                    "Context" -> "QW`"
                },
                {
                    "Documentation",
                    "Language" -> "English"
                }
            }
    |>
]




(* --- End: QW/PacletInfo.wl --- *)



(* --- Begin: QW/Kernel/NumericDTQW.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NumericDTQW*)


BeginPackage["QW`NumericDTQW`",{"QW`Utils`"}];


Unprotect@@Names["QW`NumericDTQW`*"];


ClearAll@@Names["QW`NumericDTQW`*"];


(* ::Chapter:: *)
(*Public*)


(* ::Section:: *)
(*Inicialization*)


nmInitializeDTQW::usage=FormatUsage["nmInitializeDTQW[c, p] creates the internal variables needed to simulate a DTQW, where ```c``` is the size of the coin space and ```p``` the size of the position space."];


nmMakeCoin::usage=FormatUsage["nmMakeCoin[r, \[Theta], \[Phi]] constructs a parameterized coin matrix for a quantum walk, defined by the parameters ```r```, ```\[Theta]```, and ```\[Phi]```."];


nmMakeShift::usage=FormatUsage["nmMakeShift[] constructs the shift operator for a DQWL with the given coin and position bases."];


nmMakeUnitary::usage=FormatUsage["nmMakeUnitary[] constructs the unitary matrix for a DQWL by applying the shift operator and the coin operator given by their respective functions."];


(* ::Section:: *)
(*States*)


nmVectorState::usage=FormatUsage["nmVectorState[{{a_1,c_1,p_1},...,{a_n,c_n,p_n}}] creates a vector state given by the expression '''Sum[```a_i```Ket[{```c_i```,```p_i```}],{```i```,```1```,```n```}]'''."];


nmDMatrixState::usage=FormatUsage["nmDMatrixState[{{a_1,b_1,c_1,d_1,e_1},...,{a_n,b_n,c_n,d_n,e_n}}] creates a density matrix given by the expression "]<>"\!\(\*UnderoverscriptBox[\(\[Sum]\), \(\*StyleBox[\"i\", \"TI\"] = 1\), StyleBox[\"n\", \"TI\"]]\) \!\(\*SubscriptBox[\(a\), \(i\)]\)\!\(\*TemplateBox[{RowBox[{SubscriptBox[\"b\", \"i\"], \",\", SubscriptBox[\"c\", \"i\"]}]},\n\"Ket\"]\)\!\(\*TemplateBox[{RowBox[{SubscriptBox[\"d\", \"i\"], \",\", SubscriptBox[\"e\", \"i\"]}]},\n\"Bra\"]\).";


nmValidVectorStateQ::usage=FormatUsage["nmValidVectorStateQ[state] gives '''True''' if ```state``` is a valid '''VectorState''', and '''False''' otherwise."];


nmValidDMatrixStateQ::usage=FormatUsage["nmValidDMatrixStateQ[state] gives '''True''' if ```state``` is a valid '''DMatrixState''', and '''False''' otherwise."];


nmVectorStateToArray::usage=FormatUsage["nmVectorStateToArray[state] transforms a ```state``` of '''VectorState''' into an '''Array'''."];


nmDMatrixStateToMatrix::usage=FormatUsage["nmDMatrixStateToMatrix[state] transforms a ```state``` of '''DMatrixState''' into a '''Matrix'''."];


(* ::Section:: *)
(*DTQW*)


nmDTQW::usage=FormatUsage["nmDTQW[state,n] evaluates ```n``` steps in the DTQW with initial '''VectorState''' ```state``` using the coin and shift operators created by their respective functions.
DTQW[state,n] evaluates ```n``` steps in the DTQW with initial '''DMatrixState''' ```state``` using the coin and shift operators created by their respective functions."];


nmDTQWwD::usage=FormatUsage["nmDTQWwD[state,p,n] evaluates ```n``` steps in the DTQW with initial '''VectorState''' ```state``` using the coin and shift operators created by their respective functions and with ```p``` probability of getting a phase-flip.
DTQWwD[state,p,n] evaluates ```n``` steps in the DTQW with initial '''DMatrixState''' ```state``` using the coin and shift operators created by their respective functions and with ```p``` probability of getting a phase-flip."];


(* ::Section:: *)
(*Misc*)


nmInitialRhoState::usage=FormatUsage["nmInitialRhoState[blochVector,pos] gives the initial state of the quantum walk as a density matris of the form ```\[Rho]_{*coin*}\[TensorProduct]\[Rho]_{*position*}``` where ```\[Rho]_{*coin*}``` is the density matrix created from ```blochVector``` Bloch vector, and ```\[Rho]_{*position*}``` is the density matrix '''Ket[{```pos```[[1]]}].Bra[{```pos```[[2]]}]'''."];


nmrowQW::usage=FormatUsage["nmrowQW[state,steps] returns the probability distribution at each position of the walk with initial state ```state``` after the specified number of steps ```steps```."];


nmQPascal::usage=FormatUsage["nmQPascal[state,steps] generates a table representing the evolution of the probability distribution of a DQWL on a line, showing the probabilities at the central positions after each step up to a specified number of steps."];


nmBlochVector::usage=FormatUsage["nmBlochVector[operators] calculates the Bloch vector associated with a set of operators in ```operators```, each represented by 2\[Times]2 matrices."];


(* ::Chapter:: *)
(*Private*)


Begin["`Private`"];


(* ::Section:: *)
(*Initialize*)


nmInitializeDTQW[coinSz_Integer,posSz_Integer]:=(
coinSize=coinSz;
posSize=posSz;
coinB=Transpose[{#}]&/@IdentityMatrix[coinSz];
posB=Transpose[{#}]&/@IdentityMatrix[posSz];
);


nmMakeCoin[r_,\[Theta]_,\[Phi]_]:=CoinMat={{Sqrt[r],Sqrt[1-r]Exp[I \[Theta]]},{Sqrt[1-r]Exp[I \[Theta]],-Sqrt[r]Exp[I(\[Theta]+\[Phi])]}}


nmMakeShift[]:=ShiftMat=KroneckerProduct[coinB[[1]] . coinB[[1]]\[ConjugateTranspose],Sum[posB[[i+1]] . posB[[i]]\[ConjugateTranspose],{i,1,posSize-1}]]+
KroneckerProduct[coinB[[2]] . coinB[[2]]\[ConjugateTranspose],Sum[posB[[i-1]] . posB[[i]]\[ConjugateTranspose],{i,2,posSize}]]


nmMakeUnitary[]:=UnitaryMat=ShiftMat . KroneckerProduct[CoinMat,IdentityMatrix[posSize]];


(* ::Section:: *)
(*VectorState*)


nmValidVectorStateQ[state_VectorState]:=1==Sum[Abs[v]^2,{v,state[[1,;;,1]]}]


nmVectorStateToArray[state_VectorState?ValidVectorStateQ]:=Total[#[[1]] KroneckerProduct[coinB[[#[[2]]+Round[coinSize/2]]],posB[[#[[3]]+Round[(posSize+1)/2]]]]&/@state[[1]]] (* Esto est\[AAcute] incompleto (odds, evens) *)


nmDTQW[state_VectorState,n_Integer]:=Module[
{U=ShiftMat . KroneckerProduct[CoinMat,IdentityMatrix[posSize]]},
(* Caminata cu\[AAcute]ntica discreta en el tiempo, en una l\[IAcute]nea *)
Nest[Dot[U,#]&,N[VectorStateToArray[state]],n]
]


nmDTQWwD[state_VectorState,p_?NumericQ,n_Integer]:=Module[
{U=UnitaryMat,K1,K2,rho},
(* Caminata cu\[AAcute]ntica con decoherencia calculada seg\[UAcute]n los operadores de Kraus *)
(* C\[AAcute]lculo de los operadores de Kraus *)
K1=Sqrt[p] U;
K2=Sqrt[1-p] KroneckerProduct[PauliMatrix[3],IdentityMatrix[posSize]] . U;
(* C\[AAcute]lculo de la matriz densidad *)
rho=# . #\[ConjugateTranspose]&@N[VectorStateToArray[state]];
(* C\[AAcute]lculo del canal cu\[AAcute]ntico *)
Nest[Chop[K1 . # . K1\[ConjugateTranspose]+K2 . # . K2\[ConjugateTranspose]]&,rho,n]
]


(* ::Section:: *)
(*DMatrixState*)


nmValidDMatrixStateQ[state_DMatrixState]:=1==Sum[If[(#[[2]]==#[[4]])&&(#[[3]]==#[[5]]),#[[1]],0]&@v,{v,state[[1]]}]


nmDMatrixStateToMatrix[state_DMatrixState?ValidDMatrixStateQ]:=Total[#[[1]] 
KroneckerProduct[coinB[[#[[2]]+Round[coinSize/2]]],posB[[#[[3]]+Round[(posSize+1)/2]]]] . (KroneckerProduct[coinB[[#[[4]]+Round[coinSize/2]]],posB[[#[[5]]+Round[(posSize+1)/2]]]])\[ConjugateTranspose]&/@state[[1]]
]


nmDTQW[state_DMatrixState,n_Integer]:=Module[
{U=UnitaryMat},
Nest[U . # . U\[ConjugateTranspose]&,N[DMatrixStateToMatrix[state]],n]
]


nmDTQWwD[state_DMatrixState,p_?NumericQ,n_Integer]:=Module[
{U=UnitaryMat,K1,K2,rho},
K1=Sqrt[p] U;
K2=Sqrt[1-p] KroneckerProduct[PauliMatrix[3],IdentityMatrix[posSize]] . U;
rho=N[DMatrixStateToMatrix[state]];
Nest[Chop[K1 . # . K1\[ConjugateTranspose]+K2 . # . K2\[ConjugateTranspose]]&,rho,n]
]


(* ::Section:: *)
(*Misc*)


nmInitialRhoState[blochVector_List,pos_List]:=Module[
{a,b,c,d,i,j},
{i,j}=pos;
{{a,b},{c,d}}=(IdentityMatrix[2]+blochVector . PauliMatrix[Range[3]])/2;
DMatrixState[{{a,0,i,0,j},{b,0,i,1,j},{c,1,i,0,j},{d,1,i,1,j}}]
]


nmrowQW[state_,steps_]:=Module[
{rho,prob},
rho=# . #\[ConjugateTranspose]&@DTQW[state,steps];
prob=Abs@Diagonal@MatrixPartialTrace[rho,1,{2,201}]
]


nmQPascal[state_,steps_]:=Module[
{},
TableForm[#,
TableHeadings->{Range[0,steps],Ket[{#}]&/@ Range[-steps,steps]},
TableSpacing->{2, 2},
TableAlignments->Center,
Label
]&@
(rowQW[state,#][[101-steps;;101+steps]]&/@Range[0,steps])
]


nmBlochVector[operators_List]:=Module[
{a,b,c,d,\[Sigma],r,Id,Ek},
Id=IdentityMatrix[2];
r={Cos[\[Phi]]Sin[\[Theta]],Sin[\[Phi]]Sin[\[Theta]],Cos[\[Theta]]};
\[Sigma]=PauliMatrix/@Range[3];
Ek=operators;
{{a,b},{c,d}}=Sum[#[[i]] . ((Id+r . \[Sigma])/2) . ConjugateTranspose[#[[i]]],{i,1,Length@#,1}]&@Ek;
Assuming[
\[Theta]\[Element]Reals\[And]\[Phi]\[Element]Reals\[And]p\[Element]Reals\[And]0<p<=1,
If[(FullSimplify[2a-1]==FullSimplify[-2d+1])&&(FullSimplify[2b]==FullSimplify[Conjugate[2c]]),
FullSimplify[{Re[2c],Im[2c],2a-1}],
"Error"
]]
]


End[];


Protect@@Names["QW`NumericDTQW`*"];


EndPackage[];




(* --- End: QW/Kernel/NumericDTQW.wl --- *)



(* --- Begin: QW/Kernel/RandomWalks.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*RandomWalks*)


BeginPackage["QW`RandomWalks`",{"QW`Utils`"}];


Unprotect@@Names["QW`RandomWalks`*"];


ClearAll@@Names["QW`RandomWalks`*"];


(* ::Chapter:: *)
(*Public*)


PascalRow::usage=FormatUsage["PascalRow[n] generates the ```n-```th row of Pascal's triangle interleaved with zeros. Each element is normalized by dividing by ```2^n```."];


RandomWalkDistribution::usage=FormatUsage["RandomWalkDistribution[n] returns the probability distribution associated with a quantum walker in a DTQW after ```n``` steps. The output is a list whose elements are of the form ```{p_i, x_i}```, where ```p_i``` is the probability that the walker is at position ```x_i``` after ```n``` steps."];


(* ::Chapter:: *)
(*Private*)


Begin["`Private`"];


PascalRow[n_Integer]:=Riffle[Binomial[n,#]/(2.^n)&@Range[0,n],0]


RandomWalkDistribution[n_Integer]:=Transpose[{Range[-n,n],PascalRow[n]}]


End[];


Protect@@Names["QW`RandomWalks`*"];


EndPackage[];




(* --- End: QW/Kernel/RandomWalks.wl --- *)



(* --- Begin: QW/Kernel/QW.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*QW - Package*)


BeginPackage["QW`"];


EndPackage[];


<<QW`Utils`;


<<QW`AnaliticDTQW`;


<<QW`NewDTQW`;


<<QW`NumericDTQW`;


<<QW`RandomWalks`;




(* --- End: QW/Kernel/QW.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


BeginPackage["QW`NewDTQW`",{"QW`Utils`"}];


Unprotect@@Names["QW`NewDTQW`*"];


ClearAll@@Names["QW`NewDTQW`*"];


<<`DTQWEnv`;


<<`DTQWSpacing`;


<<`DTQWApplyOperator`;


<<`DTQWStep`;


<<`DTQW`;


<<`DTQWTrace`;


<<`DTQWReducedMatrix`;


<<`DTQWPosDistribution`;


<<`DTQWPosExpvalue`;


<<`DTQWPosStandardev`;


<<`DTQWSdAdjustments`;


<<`DTQWSummary`;


<<`DTQWQtmcPrint`;


EndPackage[];




(* --- End: QW/Kernel/NewDTQW.wl --- *)



(* --- Begin: QW/Kernel/Utils/FormatUsageCase.wl --- *)


(* ::Package:: *)

FormatUsageCase;


Begin["`Private`"];


SyntaxInformation[FormatUsageCase]={"ArgumentsPattern"->{_,OptionsPattern[]}};


Options[FormatUsageCase]={StartOfLine->False};


FormatUsageCase[str_,OptionsPattern[]]:=StringReplace[
str,
(
(func:(WordCharacter|"$"|"`")../;!StringContainsQ[func,"``"])~~
args:("["~~Except["["|"]"]...~~"]")...:>
"[*"<>func<>StringReplace[args,arg:WordCharacter..:>"```"<>arg<>"```"]<>"*]"
)/.(rhs_:>lhs_):>{
If[OptionValue[StartOfLine],StartOfLine~~rhs~~w:WhitespaceCharacter:>lhs<>w,Nothing],
"[*"~~rhs~~"*]":>lhs
}
]


End[];




(* --- End: QW/Kernel/Utils/FormatUsageCase.wl --- *)



(* --- Begin: QW/Kernel/Utils/ParseFormattingDoc.wl --- *)


(* ::Package:: *)

Quiet[
ParseFormatting::usage=FormatUsage@"ParseFormatting[str] returns a box expression formatted according to the format specification of ```str```.";,
{FrontEndObject::notavail,First::normal}
];




(* --- End: QW/Kernel/Utils/ParseFormattingDoc.wl --- *)



(* --- Begin: QW/Kernel/Utils/FormatUsage.wl --- *)


(* ::Package:: *)

FormatUsage;


Begin["`Private`"];


SyntaxInformation[Unevaluated@FormatUsage]={"ArgumentsPattern"->{_}};


FormatUsage[str_]:=MakeUsageString@Map[ParseFormatting@FormatUsageCase[#,StartOfLine->True]&]@StringSplit[str,"\n"]


End[];




(* --- End: QW/Kernel/Utils/FormatUsage.wl --- *)



(* --- Begin: QW/Kernel/Utils/MakeUsageString.wl --- *)


(* ::Package:: *)

MakeUsageString;


Begin["`Private`"];


MakeUsageString[boxes_]:=MakeUsageString[{boxes}]


MakeUsageString[boxes_List]:=StringRiffle[
If[StringStartsQ[#,"\!"],#,"\!\(\)"<>#]&/@(
Replace[
boxes,
{
TagBox[b_,"[**]"]:>StyleBox[b,"MR"],
TagBox[RowBox@l:{__String},"<**>"]:>DocID[Evaluate@StringJoin@l][Label],
TagBox[b_,_]:>b
},
All      
]//Replace[
#,
s_String:>StringReplace[s,","~~EndOfString->", "],
{3}
]&//Replace[
#,
s_String?(StringContainsQ["\""]):>
"\""<>StringReplace[s,"\""->"\\\""]<>"\"",
{4,Infinity}
]&//Replace[
#,
RowBox[l_]|box_:>StringJoin@Replace[#&[l,{box}],b:Except[_String]:>
"\!\(\*"<>ToString[b,InputForm]<>"\)",1],
1
]&
),
"\n"
]


End[];




(* --- End: QW/Kernel/Utils/MakeUsageString.wl --- *)



(* --- Begin: QW/Kernel/Utils/MakeUsageStringDoc.wl --- *)


(* ::Package:: *)

Quiet[
MakeUsageString::usage=FormatUsage@"MakeUsageString[boxes] converts the box expression returned by [*ParseFormatting*] to a string that can be used as usage message.
MakeUsageString[{boxes_1,\[Ellipsis]}] creates a multiline string, with line ```i``` corresponding to ```boxes_i```.";,
{FrontEndObject::notavail,First::normal}
];




(* --- End: QW/Kernel/Utils/MakeUsageStringDoc.wl --- *)



(* --- Begin: QW/Kernel/Utils/FormatUsageDoc.wl --- *)


(* ::Package:: *)

Quiet[
FormatUsage::usage=FormatUsage@"FormatUsage[str] combines the functionalities of [*FormatUsageCase*], [*ParseFormatting*] and [*MakeUsageString*].";,
{FrontEndObject::notavail,First::normal}
];




(* --- End: QW/Kernel/Utils/FormatUsageDoc.wl --- *)



(* --- Begin: QW/Kernel/Utils/ParseFormatting.wl --- *)


(* ::Package:: *)

ParseFormatting;


Begin["`Private`ParseFormatting`"]


ti;
mr;
sc;
it;
go;
gc;
co;
cc;
lo;
lc;
sb;
sp;


ToRowBox[{el_}]:=el
ToRowBox[l_]:=RowBox@l


$closingSequences=<|ti->"```",mr->"'''",sc->"***",it->"///",gc->"*}",cc->"*]",lc->"*>"|>;


ParseToToken[str_,i_,simplify_:True][t_]:=(
  If[simplify,ToRowBox,RowBox]@Flatten@Reap[
    Module[{curToken},
      While[True,
        If[i>Length@str,Throw[$closingSequences[t],EndOfFile]];
        curToken=str[[i]];
        ++i;
        If[curToken===t,Break[]];
        Sow@ParseToken[str,i][curToken]
      ];
    ]
  ][[2]]
)
Attributes[ParseToToken]={HoldRest};


ParseToken[str_,i_][ti]:=StyleBox[ParseToToken[str,i][ti],"TI"]
ParseToken[str_,i_][mr]:=StyleBox[ParseToToken[str,i][mr],"MR"]
ParseToken[str_,i_][sc]:=StyleBox[ParseToToken[str,i][sc],ShowSpecialCharacters->False]
ParseToken[str_,i_][it]:=StyleBox[ParseToToken[str,i][it],FontSlant->Italic]
ParseToken[str_,i_][go]:=ParseToToken[str,i,False][gc]
ParseToken[str_,i_][co]:=TagBox[ParseToToken[str,i][cc],"[**]"]
ParseToken[str_,i_][lo]:=TagBox[ParseToToken[str,i][lc],"<**>"]
ParseToken[str_,i_][t:gc|cc|lc]:=Throw[$closingSequences[t],"Unmatched"]  
ParseToken[str_,i_][t_]:=t
Attributes[ParseToken]={HoldRest};


ParseFormatting::endReached="End reached while looking for `` during parsing of \"``\".";
ParseFormatting::unmatched="Unmatched closing group sequence `` found while parsing \"``\".";
ParseFormatting[str_]:=Module[
  {i=1,pStr},
  pStr=StringReplace[
    {
      "\\"~~c_:>c,
      "```"->" ```ti ",
      "'''"->" ```mr ",
      "***"->" ```sc ",
      "///"->" ```it ",
      "{*"->" ```go ",
      "*}"->" ```gc ",
      "[*"->" ```co ",
      "*]"->" ```cc ",
      "<*"->" ```lo ",
      "*>"->" ```lc ",
      ", "->" ```cm ",
      " "->" ```ws ",
      "_"->" ```sb ",
      "^"->" ```sp ",
      "\""->"```qt"
    }
  ]@str;
  pStr=StringReplace[pStr,"\\"->"\\\\"];
  pStr=First@MathLink`CallFrontEnd[
    FrontEnd`UndocumentedTestFEParserPacket[pStr,True]
  ];
  pStr=pStr/.s_String:>StringReplace[s,{"```qt"->"\"","\\\\"->"\\"}];
  pStr=Append[EndOfLine]@Replace[
    Flatten@Replace[{First@pStr},RowBox@x_:>x,\[Infinity]],
    {
      "```ti"->ti,
      "```mr"->mr,
      "```sc"->sc,
      "```it"->it,
      "```go"->go,
      "```gc"->gc,
      "```co"->co,
      "```cc"->cc,
      "```lo"->lo,
      "```lc"->lc,
      "```ws"->" ",
      "```sb"->sb,
      "```sp"->sp
    },
    1
  ];
  Catch[
    FixedPoint[
      Replace[#,{pre___,s:Longest@Repeated[_String,{2,\[Infinity]}],post___}:>
       {pre,StringReplace["```cm"->", "]@StringJoin@s,post},1]&,
      First[
        {ParseToToken[pStr, i][EndOfLine]}//.
         {pre___,a:Except[sb|sp]:"",scr:sb|sp,b:Except[sb|sp]:"",post___}:>
          {pre,If[b==="",a,If[scr===sb,SubscriptBox,SuperscriptBox][a,b]],post}
      ]
    ]/."```cm"->",",
    EndOfFile|"Unmatched",
    (
      Replace[
        {##},
        {
          {seq_,EndOfFile}:>Message[ParseFormatting::endReached,seq,str],
          {seq_,"Unmatched"}:>Message[ParseFormatting::unmatched,seq,str]
        }
      ];
      str
    )&
  ]
]
SyntaxInformation[ParseFormatting]={"ArgumentsPattern"->{_}};


End[]




(* --- End: QW/Kernel/Utils/ParseFormatting.wl --- *)



(* --- Begin: QW/Kernel/Utils/FormatUsageCaseDoc.wl --- *)


(* ::Package:: *)

Quiet[
FormatUsageCase::usage=FormatUsage@"FormatUsageCase[str] prepares all function calls wrapped in \\[* and \\*] to be formatted by [*ParseFormatting*].";,
{FrontEndObject::notavail,First::normal}
];




(* --- End: QW/Kernel/Utils/FormatUsageCaseDoc.wl --- *)



(* --- Begin: QW/Kernel/Utils.wl --- *)


(* ::Package:: *)

BeginPackage["QW`Utils`"];


Unprotect@@Names["QW`Utils`*"];


ClearAll@@Names["QW`Utils`*"];


(* ::Text:: *)
(*Definitions first*)


<<`ParseFormatting`;


<<`FormatUsageCase`;


<<`MakeUsageString`;


<<`FormatUsage`;


(* ::Text:: *)
(*Documentation after*)


<<`ParseFormattingDoc`;


<<`FormatUsageCaseDoc`;


<<`MakeUsageStringDoc`;


<<`FormatUsageDoc`;


EndPackage[];




(* --- End: QW/Kernel/Utils.wl --- *)



(* --- Begin: QW/Kernel/AnaliticDTQW.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*AnaliticDTQW*)


BeginPackage["QW`AnaliticDTQW`",{"QW`Utils`"}];


Unprotect@@Names["QW`AnaliticDTQW`*"];


ClearAll@@Names["QW`AnaliticDTQW`*"];


(* ::Chapter:: *)
(*Public*)


(* ::Section:: *)
(*Unitary Part*)


AnlCoin::usage=FormatUsage["AnlCoin[state] returns the analytic expression of coin operator applied to ```state```."];


AnlShift::usage=FormatUsage["AnlShift[state] returns the analytic expression of shift operator applied to ```state```."];


AnlDTQWstep::usage=FormatUsage["AnlDTQWstep[state] returns the analytic expression of a DTQW step of ```state```."];


(* ::Section:: *)
(*Decoherent Part*)


SigmaZ::usage=FormatUsage["SigmaZ[|c,m\[RightAngleBracket]] returns the analytic expression of ('''\[Sigma]_z''' \[CircleTimes] '''\[DoubleStruckCapitalI]''')|```c,m```\[RightAngleBracket]."];


AnlChannel::usage=FormatUsage["AnlChannel[\[Rho]_0,p] returns the analytic expression of a DTQW with phase-flip noise with probability ```p```, with initial state \[Rho]_0."];


(* ::Chapter:: *)
(*Private*)


Begin["`Private`"];


(* ::Section:: *)
(*Unitary Part*)


(* ::Text:: *)
(*Quiero entender porqu\[EAcute] aqu\[IAcute] hay bras*)


(* ::Input::Initialization:: *)
AnlCoin[state_]:=state/.{
Ket[{c_,p_}]/;c==0->(Ket[{0,p}]+Ket[{1,p}])/Sqrt[2],
Ket[{c_,p_}]/;c==1->(Ket[{0,p}]-Ket[{1,p}])/Sqrt[2],
Bra[{c_,p_}]/;c==0->(Bra[{0,p}]+Bra[{1,p}])/Sqrt[2],
Bra[{c_,p_}]/;c==1->(Bra[{0,p}]-Bra[{1,p}])/Sqrt[2]
}


(* ::Input::Initialization:: *)
AnlShift[state_]:=state/.{
Ket[{c_,p_}]/;c==0->Ket[{0,p+1}],
Ket[{c_,p_}]/;c==1->Ket[{1,p-1}],
Bra[{c_,p_}]/;c==0->Bra[{0,p+1}],
Bra[{c_,p_}]/;c==1->Bra[{1,p-1}]
}


(* ::Input::Initialization:: *)
AnlDTQWstep[state_]:=AnlShift@AnlCoin@state//FullSimplify


(* ::Section:: *)
(*Decoherent Part*)


(* ::Input::Initialization:: *)
SigmaZ[state_]:=state/.{
Ket[{c_,p_}]/;c==0->Ket[{c,p}],
Ket[{c_,p_}]/;c==1->-Ket[{c,p}],
Bra[{c_,p_}]/;c==0->Bra[{c,p}],
Bra[{c_,p_}]/;c==1->-Bra[{c,p}]
}


(* ::Input::Initialization:: *)
AnlChannel[state_,p_]:=p AnlDTQWstep[state]+(1-p) SigmaZ[AnlDTQWstep[state]]//FullSimplify//TensorExpand


End[];


Protect@@Names["QW`AnaliticDTQW`*"];


EndPackage[];




(* --- End: QW/Kernel/AnaliticDTQW.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWSdAdjustments.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWSdAdjustments*)


Quiet[
DTQWSdAdjustments::usage=FormatUsage@"DTQWSdAdjustments[stdev] Returns the best adjustment for ```stdev```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWSdAdjustments[stdev_]:=Module[{lm,nlm},
lm=NonlinearModelFit[stdev,A*x,{A},x];
nlm=NonlinearModelFit[stdev,A*Sqrt[x],{A},x];
If[lm["AdjustedRSquared"]<nlm["AdjustedRSquared"],
nlm,
lm
]
]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWSdAdjustments.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWReducedMatrix.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWReducedMatrix*)


Quiet[
DTQWReducedMatrix::usage=FormatUsage@"DTQWReducedMatrix[rho,h] Calculates the reduced density matrix of ```rho``` for either the position or coin space based on ```h```; 1 for the space, 2 for the coin.";,
{FrontEndObject::notavail,First::normal}
];


MatrixPartialTrace=ResourceFunction["MatrixPartialTrace"];


Begin["`Private`"];


DTQWReducedMatrix[rho_,h_]:=MatrixPartialTrace[rho,h,{Length[rho]/2,2}]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWReducedMatrix.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWPosStandardev.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWPosStandardev*)


Quiet[
DTQWPosStandardev::usage=FormatUsage@"DTQWPosStandardev[prob] Calculates the standard deviation from the probability distribution ```prob```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


(* ::Text:: *)
(*known issues: It should consider how the DTQW grows to know if it goes both ways or just forward.*)


DTQWPosStandardev[prob_]:=Sqrt[(Range[Length[#]]^2) . #-DTQWPosExpvalue[#]^2]&[prob]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWPosStandardev.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWSummary.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWSummary*)


Quiet[
DTQWSummary::usage=FormatUsage@"DTQWSummary[coin,n,env] Calculates the probability distribution of the last step, as well as the standard deviation for each step.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWSummary[coin_,n_,env_]:=Module[{steps,probs,sds,adjst},
steps=DTQWTrace[coin,n,env];
probs=DTQWPosDistribution/@steps;
sds=DTQWPosStandardev/@probs;
adjst=DTQWSdAdjustments[sds];
{probs[[-1]],sds}
]


DTQWSummary[coin_,env_]:=DTQWSummary[coin,100,env]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWSummary.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWApplyOperator.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWApplyOperator*)


Quiet[
DTQWApplyOperator::usage=FormatUsage@"DTQWApplyOperator[operator,rho] Uses ```operator``` to evolve the density matrix ```rho```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWApplyOperator[operator_?MatrixQ,rho_]:=operator . rho . ConjugateTranspose[operator]


DTQWApplyOperator[operator_Function,rho_]:=operator[rho]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWApplyOperator.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWEnv.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWEnvironment*)


Quiet[
DTQWEnv::usage=FormatUsage@"DTQWEnv[asoc] Creates an <*Association*> ```asoc``` that each function will use as context to perform the correct simulation.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


$DTQWEnvIcon=Graphics[{
Thick,
Circle[],
PointSize[Large],
Point[{0,0}]
},
ImageSize->Dynamic[{
Automatic,
3.5 CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
}]
];


DTQWEnvAscQ[asc_?AssociationQ]:=And[
AllTrue[{"Type","Decoherent"},KeyExistsQ[asc,#]&],
Or[KeyExistsQ[asc,"Unitary"],AllTrue[{"Shift","Coin"},KeyExistsQ[asc,#]&]],
Xnor[asc["Decoherent"],AllTrue[{"Kraus","Probabilities"},KeyExistsQ[asc,#]&]],
Xnor[asc["Type"]=="FixedSize",KeyExistsQ[asc,"Size"]]
]


DTQWEnv[asc_?DTQWEnvAscQ]:=Module[{shift,coin,unit},
shift=asc["Shift"];
coin=asc["Coin"];
unit=Function[n,shift[n] . coin[n]];
DTQWEnv[Append[asc,"Unitary"->unit]]
]/;Not[KeyExistsQ[asc,"Unitary"]]


DTQWEnv/:MakeBoxes[obj:DTQWEnv[asc_?DTQWEnvAscQ],form:(StandardForm|TraditionalForm)]:=
Module[{above,below},
above={
{BoxForm`SummaryItem[{"Type: ",asc["Type"]}],SpanFromLeft},
{BoxForm`SummaryItem[{"Decoherent: ",asc["Decoherent"]}],BoxForm`SummaryItem[{"Size: ",If[KeyExistsQ[asc,"Size"],asc["Size"],{\[Infinity],2}]}]}
};
below={
{BoxForm`SummaryItem[{"Information: ","This object represents a 1-dimensional discrete-time quantum walk with a two-dimensional coin"}],SpanFromLeft}
};
BoxForm`ArrangeSummaryBox[
DTQWEnv,
obj,
$DTQWEnvIcon,
above,
below,
form,
"Interpretable"->Automatic
]
];


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWEnv.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQW.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQW*)


Quiet[
DTQW::usage=FormatUsage@"DTQW[coin,n,env] Executes ```n``` steps of the DTQW defined by ```env``` taking ```coin``` as the initial state for the coin with position 0.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQW[c0_,n_,env_]:=Module[{\[Rho]},
\[Rho]=Outer[Times,#,Conjugate[#]]&[c0];
Do[\[Rho]=DTQWStep[\[Rho],env],{t,n}];
\[Rho]
]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQW.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWStep.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWStep*)


Quiet[
DTQWStep::usage=FormatUsage@"DTQWStep[rho,env] Makes one step of the DTQW defined by the ```env``` with inital state ```rho```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWStep[rho_,DTQWEnv[env_]]:=With[{
bigrho=DTQWSpacing[rho,env["Type"]]
},
Chop[DTQWApplyOperator[#,bigrho]]&[env["Unitary"][Length[bigrho]/2]]/;Not[env["Decoherent"]]
]


DTQWStep[rho_,DTQWEnv[env_]]:=Module[{
bigrho=DTQWSpacing[rho,env["Type"]],
state
},
state=DTQWApplyOperator[#,bigrho]&[env["Unitary"][Length[bigrho]/2]];
Inner[Chop[#1*DTQWApplyOperator[#2[Length[bigrho]/2],state]]&,env["Probabilities"],env["Kraus"],Plus]
]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWStep.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWTrace.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWTrace*)


Quiet[
DTQWTrace::usage=FormatUsage@"DTQWTrace[coin,n,env] Executes ```n``` steps of the DTQW defined by ```env``` taking ```coin``` as the initial state for the coin with position 0 and returns every step until the end.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWTrace[c0_,n_,env_]:=Module[{\[Rho]},
\[Rho]=Outer[Times,#,Conjugate[#]]&[c0];
Table[\[Rho]=DTQWStep[\[Rho],env],{t,n}]
]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWTrace.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWPosExpvalue.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWPosExpvalue*)


Quiet[
DTQWPosExpvalue::usage=FormatUsage@"DTQWPosExpvalue[prob] Calculates the expected value from the probability distribution ```prob```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


(* ::Text:: *)
(*known issues: It should consider how the DTQW grows to know if it goes both ways or just forward.*)


DTQWPosExpvalue[prob_]:=Range[Length[#]] . #&[prob]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWPosExpvalue.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWPosDistribution.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWPosDistribution*)


Quiet[
DTQWPosDistribution::usage=FormatUsage@"DTQWPosDistribution[rho] Gets the probability distribution for the position for a given ```rho```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWPosDistribution[rho_]:=Total/@Partition[Diagonal@rho,2]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWPosDistribution.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWQtmcPrint.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWQtmcPrint*)


Quiet[
DTQWQtmcPrint::usage=FormatUsage@"DTQWQtmcPrint[probs,kraus] Prints an quantum channel with probabilities ```probs``` and kraus operators ```kraus```, it relies on an external API.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


(* ::Text:: *)
(*Known issues: It lacks of more flexibility to represent a bigger variety of quantum channels (also should have a way to change API in any case one is down).*)


DTQWQtmcPrint[probs_,kraus_]:=With[{tex="\\mathcal{E}(\\rho) = "<>Fold[#1<>"+"<>#2&,MapThread[#1<>""<>#2[[1]]<>"U \\rho U^\\dagger "<>#2[[2]]&,{probs,Map[If[""==#,{"",""},{#,#<>"^\\dagger "}]&,kraus]}]]},
URLExecute["https://math.vercel.app/",{"bgcolor"->"auto","from"->"\\LARGE "<>ToString[tex]<>".svg"}]//Style[#,TextAlignment->Center]&
]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWQtmcPrint.wl --- *)



(* --- Begin: QW/Kernel/NewDTQW/DTQWSpacing.wl --- *)


(* ::Package:: *)

(* ::Title:: *)
(*NewDTQW - Package*)


(* ::Subtitle:: *)
(*DTQWSpacing*)


Quiet[
DTQWSpacing::usage=FormatUsage@"DTQWSpacing[rho,spec] Recives a density matrix ```rho``` and grows its size depending on ```spec```.";,
{FrontEndObject::notavail,First::normal}
];


Begin["`Private`"];


DTQWSpacing[rho_,"FixedSize"]:=rho


DTQWSpacing[rho_,"Growing"]:=ArrayPad[rho,2]


DTQWSpacing[rho_,"ForwardGrowing"]:=ArrayPad[rho,{{0,2},{0,2}}]


DTQWSpacing[rho_,s_String]:=With[{n=Read[StringToStream[StringDrop[s,-5]],Number]},
ArrayPad[rho,{{0,2*n},{0,2*n}}]
]/;(StringTake[s,-5]=="Ahead")


DTQWSpacing[rho_,"Both"]:=ArrayPad[rho,2]


End[];




(* --- End: QW/Kernel/NewDTQW/DTQWSpacing.wl --- *)


