��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.22v2.9.1-132-g18960c44ad38��
�
 SGD/y1_predictions/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" SGD/y1_predictions/bias/momentum
�
4SGD/y1_predictions/bias/momentum/Read/ReadVariableOpReadVariableOp SGD/y1_predictions/bias/momentum*
_output_shapes
:*
dtype0
�
"SGD/y1_predictions/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*3
shared_name$"SGD/y1_predictions/kernel/momentum
�
6SGD/y1_predictions/kernel/momentum/Read/ReadVariableOpReadVariableOp"SGD/y1_predictions/kernel/momentum*
_output_shapes

:d*
dtype0
�
 SGD/y0_predictions/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" SGD/y0_predictions/bias/momentum
�
4SGD/y0_predictions/bias/momentum/Read/ReadVariableOpReadVariableOp SGD/y0_predictions/bias/momentum*
_output_shapes
:*
dtype0
�
"SGD/y0_predictions/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*3
shared_name$"SGD/y0_predictions/kernel/momentum
�
6SGD/y0_predictions/kernel/momentum/Read/ReadVariableOpReadVariableOp"SGD/y0_predictions/kernel/momentum*
_output_shapes

:d*
dtype0
�
SGD/y1_hidden_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameSGD/y1_hidden_2/bias/momentum
�
1SGD/y1_hidden_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/y1_hidden_2/bias/momentum*
_output_shapes
:d*
dtype0
�
SGD/y1_hidden_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*0
shared_name!SGD/y1_hidden_2/kernel/momentum
�
3SGD/y1_hidden_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/y1_hidden_2/kernel/momentum*
_output_shapes

:dd*
dtype0
�
SGD/y0_hidden_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameSGD/y0_hidden_2/bias/momentum
�
1SGD/y0_hidden_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/y0_hidden_2/bias/momentum*
_output_shapes
:d*
dtype0
�
SGD/y0_hidden_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*0
shared_name!SGD/y0_hidden_2/kernel/momentum
�
3SGD/y0_hidden_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/y0_hidden_2/kernel/momentum*
_output_shapes

:dd*
dtype0
�
SGD/y1_hidden_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameSGD/y1_hidden_1/bias/momentum
�
1SGD/y1_hidden_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/y1_hidden_1/bias/momentum*
_output_shapes
:d*
dtype0
�
SGD/y1_hidden_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*0
shared_name!SGD/y1_hidden_1/kernel/momentum
�
3SGD/y1_hidden_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/y1_hidden_1/kernel/momentum*
_output_shapes
:	�d*
dtype0
�
SGD/y0_hidden_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameSGD/y0_hidden_1/bias/momentum
�
1SGD/y0_hidden_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/y0_hidden_1/bias/momentum*
_output_shapes
:d*
dtype0
�
SGD/y0_hidden_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*0
shared_name!SGD/y0_hidden_1/kernel/momentum
�
3SGD/y0_hidden_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/y0_hidden_1/kernel/momentum*
_output_shapes
:	�d*
dtype0
�
SGD/phi_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameSGD/phi_3/bias/momentum
�
+SGD/phi_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/phi_3/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGD/phi_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_nameSGD/phi_3/kernel/momentum
�
-SGD/phi_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/phi_3/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
SGD/phi_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameSGD/phi_2/bias/momentum
�
+SGD/phi_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/phi_2/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGD/phi_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_nameSGD/phi_2/kernel/momentum
�
-SGD/phi_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/phi_2/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
SGD/phi_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameSGD/phi_1/bias/momentum
�
+SGD/phi_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/phi_1/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGD/phi_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_nameSGD/phi_1/kernel/momentum
�
-SGD/phi_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/phi_1/kernel/momentum*
_output_shapes
:	�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
~
y1_predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namey1_predictions/bias
w
'y1_predictions/bias/Read/ReadVariableOpReadVariableOpy1_predictions/bias*
_output_shapes
:*
dtype0
�
y1_predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_namey1_predictions/kernel

)y1_predictions/kernel/Read/ReadVariableOpReadVariableOpy1_predictions/kernel*
_output_shapes

:d*
dtype0
~
y0_predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namey0_predictions/bias
w
'y0_predictions/bias/Read/ReadVariableOpReadVariableOpy0_predictions/bias*
_output_shapes
:*
dtype0
�
y0_predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_namey0_predictions/kernel

)y0_predictions/kernel/Read/ReadVariableOpReadVariableOpy0_predictions/kernel*
_output_shapes

:d*
dtype0
x
y1_hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_namey1_hidden_2/bias
q
$y1_hidden_2/bias/Read/ReadVariableOpReadVariableOpy1_hidden_2/bias*
_output_shapes
:d*
dtype0
�
y1_hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*#
shared_namey1_hidden_2/kernel
y
&y1_hidden_2/kernel/Read/ReadVariableOpReadVariableOpy1_hidden_2/kernel*
_output_shapes

:dd*
dtype0
x
y0_hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_namey0_hidden_2/bias
q
$y0_hidden_2/bias/Read/ReadVariableOpReadVariableOpy0_hidden_2/bias*
_output_shapes
:d*
dtype0
�
y0_hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*#
shared_namey0_hidden_2/kernel
y
&y0_hidden_2/kernel/Read/ReadVariableOpReadVariableOpy0_hidden_2/kernel*
_output_shapes

:dd*
dtype0
x
y1_hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_namey1_hidden_1/bias
q
$y1_hidden_1/bias/Read/ReadVariableOpReadVariableOpy1_hidden_1/bias*
_output_shapes
:d*
dtype0
�
y1_hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*#
shared_namey1_hidden_1/kernel
z
&y1_hidden_1/kernel/Read/ReadVariableOpReadVariableOpy1_hidden_1/kernel*
_output_shapes
:	�d*
dtype0
x
y0_hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_namey0_hidden_1/bias
q
$y0_hidden_1/bias/Read/ReadVariableOpReadVariableOpy0_hidden_1/bias*
_output_shapes
:d*
dtype0
�
y0_hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*#
shared_namey0_hidden_1/kernel
z
&y0_hidden_1/kernel/Read/ReadVariableOpReadVariableOpy0_hidden_1/kernel*
_output_shapes
:	�d*
dtype0
m

phi_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
phi_3/bias
f
phi_3/bias/Read/ReadVariableOpReadVariableOp
phi_3/bias*
_output_shapes	
:�*
dtype0
v
phi_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namephi_3/kernel
o
 phi_3/kernel/Read/ReadVariableOpReadVariableOpphi_3/kernel* 
_output_shapes
:
��*
dtype0
m

phi_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
phi_2/bias
f
phi_2/bias/Read/ReadVariableOpReadVariableOp
phi_2/bias*
_output_shapes	
:�*
dtype0
v
phi_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namephi_2/kernel
o
 phi_2/kernel/Read/ReadVariableOpReadVariableOpphi_2/kernel* 
_output_shapes
:
��*
dtype0
m

phi_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
phi_1/bias
f
phi_1/bias/Read/ReadVariableOpReadVariableOp
phi_1/bias*
_output_shapes	
:�*
dtype0
u
phi_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namephi_1/kernel
n
 phi_1/kernel/Read/ReadVariableOpReadVariableOpphi_1/kernel*
_output_shapes
:	�*
dtype0

NoOpNoOp
�e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�e
value�eB�e B�e
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17*
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17*
,
c0
d1
e2
f3
g4
h5* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ntrace_0
otrace_1
ptrace_2
qtrace_3* 
6
rtrace_0
strace_1
ttrace_2
utrace_3* 
* 
�
viter
	wdecay
xlearning_rate
ymomentummomentum�momentum�#momentum�$momentum�+momentum�,momentum�3momentum�4momentum�;momentum�<momentum�Cmomentum�Dmomentum�Kmomentum�Lmomentum�Smomentum�Tmomentum�[momentum�\momentum�*

zserving_default* 

0
1*

0
1*
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEphi_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
phi_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEphi_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
phi_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEphi_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
phi_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
	
c0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEy0_hidden_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEy0_hidden_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
	
d0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEy1_hidden_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEy1_hidden_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
	
e0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEy0_hidden_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEy0_hidden_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

K0
L1*

K0
L1*
	
f0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEy1_hidden_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEy1_hidden_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*
	
g0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEy0_predictions/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEy0_predictions/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

[0
\1*
	
h0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEy1_predictions/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEy1_predictions/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
c0* 
* 
* 
* 
* 
* 
* 
	
d0* 
* 
* 
* 
* 
* 
* 
	
e0* 
* 
* 
* 
* 
* 
* 
	
f0* 
* 
* 
* 
* 
* 
* 
	
g0* 
* 
* 
* 
* 
* 
* 
	
h0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUESGD/phi_1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/phi_1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/phi_2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/phi_2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/phi_3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/phi_3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y0_hidden_1/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y0_hidden_1/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y1_hidden_1/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y1_hidden_1/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y0_hidden_2/kernel/momentumYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y0_hidden_2/bias/momentumWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y1_hidden_2/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/y1_hidden_2/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"SGD/y0_predictions/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE SGD/y0_predictions/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"SGD/y1_predictions/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE SGD/y1_predictions/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
x
serving_default_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputphi_1/kernel
phi_1/biasphi_2/kernel
phi_2/biasphi_3/kernel
phi_3/biasy1_hidden_1/kernely1_hidden_1/biasy0_hidden_1/kernely0_hidden_1/biasy1_hidden_2/kernely1_hidden_2/biasy0_hidden_2/kernely0_hidden_2/biasy0_predictions/kernely0_predictions/biasy1_predictions/kernely1_predictions/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_17498
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename phi_1/kernel/Read/ReadVariableOpphi_1/bias/Read/ReadVariableOp phi_2/kernel/Read/ReadVariableOpphi_2/bias/Read/ReadVariableOp phi_3/kernel/Read/ReadVariableOpphi_3/bias/Read/ReadVariableOp&y0_hidden_1/kernel/Read/ReadVariableOp$y0_hidden_1/bias/Read/ReadVariableOp&y1_hidden_1/kernel/Read/ReadVariableOp$y1_hidden_1/bias/Read/ReadVariableOp&y0_hidden_2/kernel/Read/ReadVariableOp$y0_hidden_2/bias/Read/ReadVariableOp&y1_hidden_2/kernel/Read/ReadVariableOp$y1_hidden_2/bias/Read/ReadVariableOp)y0_predictions/kernel/Read/ReadVariableOp'y0_predictions/bias/Read/ReadVariableOp)y1_predictions/kernel/Read/ReadVariableOp'y1_predictions/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-SGD/phi_1/kernel/momentum/Read/ReadVariableOp+SGD/phi_1/bias/momentum/Read/ReadVariableOp-SGD/phi_2/kernel/momentum/Read/ReadVariableOp+SGD/phi_2/bias/momentum/Read/ReadVariableOp-SGD/phi_3/kernel/momentum/Read/ReadVariableOp+SGD/phi_3/bias/momentum/Read/ReadVariableOp3SGD/y0_hidden_1/kernel/momentum/Read/ReadVariableOp1SGD/y0_hidden_1/bias/momentum/Read/ReadVariableOp3SGD/y1_hidden_1/kernel/momentum/Read/ReadVariableOp1SGD/y1_hidden_1/bias/momentum/Read/ReadVariableOp3SGD/y0_hidden_2/kernel/momentum/Read/ReadVariableOp1SGD/y0_hidden_2/bias/momentum/Read/ReadVariableOp3SGD/y1_hidden_2/kernel/momentum/Read/ReadVariableOp1SGD/y1_hidden_2/bias/momentum/Read/ReadVariableOp6SGD/y0_predictions/kernel/momentum/Read/ReadVariableOp4SGD/y0_predictions/bias/momentum/Read/ReadVariableOp6SGD/y1_predictions/kernel/momentum/Read/ReadVariableOp4SGD/y1_predictions/bias/momentum/Read/ReadVariableOpConst*9
Tin2
02.	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_18270
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamephi_1/kernel
phi_1/biasphi_2/kernel
phi_2/biasphi_3/kernel
phi_3/biasy0_hidden_1/kernely0_hidden_1/biasy1_hidden_1/kernely1_hidden_1/biasy0_hidden_2/kernely0_hidden_2/biasy1_hidden_2/kernely1_hidden_2/biasy0_predictions/kernely0_predictions/biasy1_predictions/kernely1_predictions/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcountSGD/phi_1/kernel/momentumSGD/phi_1/bias/momentumSGD/phi_2/kernel/momentumSGD/phi_2/bias/momentumSGD/phi_3/kernel/momentumSGD/phi_3/bias/momentumSGD/y0_hidden_1/kernel/momentumSGD/y0_hidden_1/bias/momentumSGD/y1_hidden_1/kernel/momentumSGD/y1_hidden_1/bias/momentumSGD/y0_hidden_2/kernel/momentumSGD/y0_hidden_2/bias/momentumSGD/y1_hidden_2/kernel/momentumSGD/y1_hidden_2/bias/momentum"SGD/y0_predictions/kernel/momentum SGD/y0_predictions/bias/momentum"SGD/y1_predictions/kernel/momentum SGD/y1_predictions/bias/momentum*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_18412��
�
�
__inference_loss_fn_4_18104R
@y0_predictions_kernel_regularizer_square_readvariableop_resource:d
identity��7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@y0_predictions_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentity)y0_predictions/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp
�
�
%__inference_phi_1_layer_call_fn_17831

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_1_layer_call_and_return_conditional_losses_16676p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_16756

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�e
�

@__inference_model_layer_call_and_return_conditional_losses_17329	
input
phi_1_17246:	�
phi_1_17248:	�
phi_2_17251:
��
phi_2_17253:	�
phi_3_17256:
��
phi_3_17258:	�$
y1_hidden_1_17261:	�d
y1_hidden_1_17263:d$
y0_hidden_1_17266:	�d
y0_hidden_1_17268:d#
y1_hidden_2_17271:dd
y1_hidden_2_17273:d#
y0_hidden_2_17276:dd
y0_hidden_2_17278:d&
y0_predictions_17281:d"
y0_predictions_17283:&
y1_predictions_17286:d"
y1_predictions_17288:
identity��phi_1/StatefulPartitionedCall�phi_2/StatefulPartitionedCall�phi_3/StatefulPartitionedCall�#y0_hidden_1/StatefulPartitionedCall�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y0_hidden_2/StatefulPartitionedCall�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y0_predictions/StatefulPartitionedCall�7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_1/StatefulPartitionedCall�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_2/StatefulPartitionedCall�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y1_predictions/StatefulPartitionedCall�7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
phi_1/StatefulPartitionedCallStatefulPartitionedCallinputphi_1_17246phi_1_17248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_1_layer_call_and_return_conditional_losses_16676�
phi_2/StatefulPartitionedCallStatefulPartitionedCall&phi_1/StatefulPartitionedCall:output:0phi_2_17251phi_2_17253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_2_layer_call_and_return_conditional_losses_16693�
phi_3/StatefulPartitionedCallStatefulPartitionedCall&phi_2/StatefulPartitionedCall:output:0phi_3_17256phi_3_17258*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_3_layer_call_and_return_conditional_losses_16710�
#y1_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y1_hidden_1_17261y1_hidden_1_17263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_16733�
#y0_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y0_hidden_1_17266y0_hidden_1_17268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_16756�
#y1_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_1/StatefulPartitionedCall:output:0y1_hidden_2_17271y1_hidden_2_17273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_16779�
#y0_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_1/StatefulPartitionedCall:output:0y0_hidden_2_17276y0_hidden_2_17278*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_16802�
&y0_predictions/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_2/StatefulPartitionedCall:output:0y0_predictions_17281y0_predictions_17283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y0_predictions_layer_call_and_return_conditional_losses_16824�
&y1_predictions/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_2/StatefulPartitionedCall:output:0y1_predictions_17286y1_predictions_17288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y1_predictions_layer_call_and_return_conditional_losses_16846�
concatenate/PartitionedCallPartitionedCall/y0_predictions/StatefulPartitionedCall:output:0/y1_predictions/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16859�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_1_17266*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_1_17261*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_2_17276*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_2_17271*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_predictions_17281*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_predictions_17286*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^phi_1/StatefulPartitionedCall^phi_2/StatefulPartitionedCall^phi_3/StatefulPartitionedCall$^y0_hidden_1/StatefulPartitionedCall5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y0_hidden_2/StatefulPartitionedCall5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y0_predictions/StatefulPartitionedCall8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_1/StatefulPartitionedCall5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_2/StatefulPartitionedCall5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y1_predictions/StatefulPartitionedCall8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2>
phi_1/StatefulPartitionedCallphi_1/StatefulPartitionedCall2>
phi_2/StatefulPartitionedCallphi_2/StatefulPartitionedCall2>
phi_3/StatefulPartitionedCallphi_3/StatefulPartitionedCall2J
#y0_hidden_1/StatefulPartitionedCall#y0_hidden_1/StatefulPartitionedCall2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y0_hidden_2/StatefulPartitionedCall#y0_hidden_2/StatefulPartitionedCall2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y0_predictions/StatefulPartitionedCall&y0_predictions/StatefulPartitionedCall2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_1/StatefulPartitionedCall#y1_hidden_1/StatefulPartitionedCall2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_2/StatefulPartitionedCall#y1_hidden_2/StatefulPartitionedCall2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y1_predictions/StatefulPartitionedCall&y1_predictions/StatefulPartitionedCall2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
I__inference_y1_predictions_layer_call_and_return_conditional_losses_16846

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�7y1_predictions/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
%__inference_phi_2_layer_call_fn_17851

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_2_layer_call_and_return_conditional_losses_16693p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_18060P
=y0_hidden_1_kernel_regularizer_square_readvariableop_resource:	�d
identity��4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=y0_hidden_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&y0_hidden_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp
�

�
@__inference_phi_2_layer_call_and_return_conditional_losses_16693

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_16937	
input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:	�d
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:d

unknown_14:

unknown_15:d

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_16898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�e
�

@__inference_model_layer_call_and_return_conditional_losses_17163

inputs
phi_1_17080:	�
phi_1_17082:	�
phi_2_17085:
��
phi_2_17087:	�
phi_3_17090:
��
phi_3_17092:	�$
y1_hidden_1_17095:	�d
y1_hidden_1_17097:d$
y0_hidden_1_17100:	�d
y0_hidden_1_17102:d#
y1_hidden_2_17105:dd
y1_hidden_2_17107:d#
y0_hidden_2_17110:dd
y0_hidden_2_17112:d&
y0_predictions_17115:d"
y0_predictions_17117:&
y1_predictions_17120:d"
y1_predictions_17122:
identity��phi_1/StatefulPartitionedCall�phi_2/StatefulPartitionedCall�phi_3/StatefulPartitionedCall�#y0_hidden_1/StatefulPartitionedCall�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y0_hidden_2/StatefulPartitionedCall�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y0_predictions/StatefulPartitionedCall�7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_1/StatefulPartitionedCall�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_2/StatefulPartitionedCall�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y1_predictions/StatefulPartitionedCall�7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
phi_1/StatefulPartitionedCallStatefulPartitionedCallinputsphi_1_17080phi_1_17082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_1_layer_call_and_return_conditional_losses_16676�
phi_2/StatefulPartitionedCallStatefulPartitionedCall&phi_1/StatefulPartitionedCall:output:0phi_2_17085phi_2_17087*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_2_layer_call_and_return_conditional_losses_16693�
phi_3/StatefulPartitionedCallStatefulPartitionedCall&phi_2/StatefulPartitionedCall:output:0phi_3_17090phi_3_17092*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_3_layer_call_and_return_conditional_losses_16710�
#y1_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y1_hidden_1_17095y1_hidden_1_17097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_16733�
#y0_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y0_hidden_1_17100y0_hidden_1_17102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_16756�
#y1_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_1/StatefulPartitionedCall:output:0y1_hidden_2_17105y1_hidden_2_17107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_16779�
#y0_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_1/StatefulPartitionedCall:output:0y0_hidden_2_17110y0_hidden_2_17112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_16802�
&y0_predictions/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_2/StatefulPartitionedCall:output:0y0_predictions_17115y0_predictions_17117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y0_predictions_layer_call_and_return_conditional_losses_16824�
&y1_predictions/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_2/StatefulPartitionedCall:output:0y1_predictions_17120y1_predictions_17122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y1_predictions_layer_call_and_return_conditional_losses_16846�
concatenate/PartitionedCallPartitionedCall/y0_predictions/StatefulPartitionedCall:output:0/y1_predictions/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16859�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_1_17100*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_1_17095*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_2_17110*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_2_17105*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_predictions_17115*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_predictions_17120*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^phi_1/StatefulPartitionedCall^phi_2/StatefulPartitionedCall^phi_3/StatefulPartitionedCall$^y0_hidden_1/StatefulPartitionedCall5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y0_hidden_2/StatefulPartitionedCall5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y0_predictions/StatefulPartitionedCall8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_1/StatefulPartitionedCall5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_2/StatefulPartitionedCall5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y1_predictions/StatefulPartitionedCall8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2>
phi_1/StatefulPartitionedCallphi_1/StatefulPartitionedCall2>
phi_2/StatefulPartitionedCallphi_2/StatefulPartitionedCall2>
phi_3/StatefulPartitionedCallphi_3/StatefulPartitionedCall2J
#y0_hidden_1/StatefulPartitionedCall#y0_hidden_1/StatefulPartitionedCall2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y0_hidden_2/StatefulPartitionedCall#y0_hidden_2/StatefulPartitionedCall2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y0_predictions/StatefulPartitionedCall&y0_predictions/StatefulPartitionedCall2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_1/StatefulPartitionedCall#y1_hidden_1/StatefulPartitionedCall2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_2/StatefulPartitionedCall#y1_hidden_2/StatefulPartitionedCall2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y1_predictions/StatefulPartitionedCall&y1_predictions/StatefulPartitionedCall2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_y1_hidden_1_layer_call_fn_17917

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_16733o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_y0_hidden_2_layer_call_fn_17943

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_16802o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
I__inference_y0_predictions_layer_call_and_return_conditional_losses_16824

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�7y0_predictions/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_16733

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_17719

inputs7
$phi_1_matmul_readvariableop_resource:	�4
%phi_1_biasadd_readvariableop_resource:	�8
$phi_2_matmul_readvariableop_resource:
��4
%phi_2_biasadd_readvariableop_resource:	�8
$phi_3_matmul_readvariableop_resource:
��4
%phi_3_biasadd_readvariableop_resource:	�=
*y1_hidden_1_matmul_readvariableop_resource:	�d9
+y1_hidden_1_biasadd_readvariableop_resource:d=
*y0_hidden_1_matmul_readvariableop_resource:	�d9
+y0_hidden_1_biasadd_readvariableop_resource:d<
*y1_hidden_2_matmul_readvariableop_resource:dd9
+y1_hidden_2_biasadd_readvariableop_resource:d<
*y0_hidden_2_matmul_readvariableop_resource:dd9
+y0_hidden_2_biasadd_readvariableop_resource:d?
-y0_predictions_matmul_readvariableop_resource:d<
.y0_predictions_biasadd_readvariableop_resource:?
-y1_predictions_matmul_readvariableop_resource:d<
.y1_predictions_biasadd_readvariableop_resource:
identity��phi_1/BiasAdd/ReadVariableOp�phi_1/MatMul/ReadVariableOp�phi_2/BiasAdd/ReadVariableOp�phi_2/MatMul/ReadVariableOp�phi_3/BiasAdd/ReadVariableOp�phi_3/MatMul/ReadVariableOp�"y0_hidden_1/BiasAdd/ReadVariableOp�!y0_hidden_1/MatMul/ReadVariableOp�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�"y0_hidden_2/BiasAdd/ReadVariableOp�!y0_hidden_2/MatMul/ReadVariableOp�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�%y0_predictions/BiasAdd/ReadVariableOp�$y0_predictions/MatMul/ReadVariableOp�7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�"y1_hidden_1/BiasAdd/ReadVariableOp�!y1_hidden_1/MatMul/ReadVariableOp�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�"y1_hidden_2/BiasAdd/ReadVariableOp�!y1_hidden_2/MatMul/ReadVariableOp�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�%y1_predictions/BiasAdd/ReadVariableOp�$y1_predictions/MatMul/ReadVariableOp�7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
phi_1/MatMul/ReadVariableOpReadVariableOp$phi_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0v
phi_1/MatMulMatMulinputs#phi_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
phi_1/BiasAdd/ReadVariableOpReadVariableOp%phi_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
phi_1/BiasAddBiasAddphi_1/MatMul:product:0$phi_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
	phi_1/EluEluphi_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
phi_2/MatMul/ReadVariableOpReadVariableOp$phi_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
phi_2/MatMulMatMulphi_1/Elu:activations:0#phi_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
phi_2/BiasAdd/ReadVariableOpReadVariableOp%phi_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
phi_2/BiasAddBiasAddphi_2/MatMul:product:0$phi_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
	phi_2/EluEluphi_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
phi_3/MatMul/ReadVariableOpReadVariableOp$phi_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
phi_3/MatMulMatMulphi_2/Elu:activations:0#phi_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
phi_3/BiasAdd/ReadVariableOpReadVariableOp%phi_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
phi_3/BiasAddBiasAddphi_3/MatMul:product:0$phi_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
	phi_3/EluEluphi_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!y1_hidden_1/MatMul/ReadVariableOpReadVariableOp*y1_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
y1_hidden_1/MatMulMatMulphi_3/Elu:activations:0)y1_hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y1_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+y1_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y1_hidden_1/BiasAddBiasAddy1_hidden_1/MatMul:product:0*y1_hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y1_hidden_1/EluEluy1_hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
!y0_hidden_1/MatMul/ReadVariableOpReadVariableOp*y0_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
y0_hidden_1/MatMulMatMulphi_3/Elu:activations:0)y0_hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y0_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+y0_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y0_hidden_1/BiasAddBiasAddy0_hidden_1/MatMul:product:0*y0_hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y0_hidden_1/EluEluy0_hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
!y1_hidden_2/MatMul/ReadVariableOpReadVariableOp*y1_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
y1_hidden_2/MatMulMatMuly1_hidden_1/Elu:activations:0)y1_hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y1_hidden_2/BiasAdd/ReadVariableOpReadVariableOp+y1_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y1_hidden_2/BiasAddBiasAddy1_hidden_2/MatMul:product:0*y1_hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y1_hidden_2/EluEluy1_hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
!y0_hidden_2/MatMul/ReadVariableOpReadVariableOp*y0_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
y0_hidden_2/MatMulMatMuly0_hidden_1/Elu:activations:0)y0_hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y0_hidden_2/BiasAdd/ReadVariableOpReadVariableOp+y0_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y0_hidden_2/BiasAddBiasAddy0_hidden_2/MatMul:product:0*y0_hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y0_hidden_2/EluEluy0_hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
$y0_predictions/MatMul/ReadVariableOpReadVariableOp-y0_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
y0_predictions/MatMulMatMuly0_hidden_2/Elu:activations:0,y0_predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%y0_predictions/BiasAdd/ReadVariableOpReadVariableOp.y0_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
y0_predictions/BiasAddBiasAddy0_predictions/MatMul:product:0-y0_predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$y1_predictions/MatMul/ReadVariableOpReadVariableOp-y1_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
y1_predictions/MatMulMatMuly1_hidden_2/Elu:activations:0,y1_predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%y1_predictions/BiasAdd/ReadVariableOpReadVariableOp.y1_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
y1_predictions/BiasAddBiasAddy1_predictions/MatMul:product:0-y1_predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2y0_predictions/BiasAdd:output:0y1_predictions/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y0_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y1_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y0_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y1_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-y0_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-y1_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityconcatenate/concat:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^phi_1/BiasAdd/ReadVariableOp^phi_1/MatMul/ReadVariableOp^phi_2/BiasAdd/ReadVariableOp^phi_2/MatMul/ReadVariableOp^phi_3/BiasAdd/ReadVariableOp^phi_3/MatMul/ReadVariableOp#^y0_hidden_1/BiasAdd/ReadVariableOp"^y0_hidden_1/MatMul/ReadVariableOp5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp#^y0_hidden_2/BiasAdd/ReadVariableOp"^y0_hidden_2/MatMul/ReadVariableOp5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp&^y0_predictions/BiasAdd/ReadVariableOp%^y0_predictions/MatMul/ReadVariableOp8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp#^y1_hidden_1/BiasAdd/ReadVariableOp"^y1_hidden_1/MatMul/ReadVariableOp5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp#^y1_hidden_2/BiasAdd/ReadVariableOp"^y1_hidden_2/MatMul/ReadVariableOp5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp&^y1_predictions/BiasAdd/ReadVariableOp%^y1_predictions/MatMul/ReadVariableOp8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2<
phi_1/BiasAdd/ReadVariableOpphi_1/BiasAdd/ReadVariableOp2:
phi_1/MatMul/ReadVariableOpphi_1/MatMul/ReadVariableOp2<
phi_2/BiasAdd/ReadVariableOpphi_2/BiasAdd/ReadVariableOp2:
phi_2/MatMul/ReadVariableOpphi_2/MatMul/ReadVariableOp2<
phi_3/BiasAdd/ReadVariableOpphi_3/BiasAdd/ReadVariableOp2:
phi_3/MatMul/ReadVariableOpphi_3/MatMul/ReadVariableOp2H
"y0_hidden_1/BiasAdd/ReadVariableOp"y0_hidden_1/BiasAdd/ReadVariableOp2F
!y0_hidden_1/MatMul/ReadVariableOp!y0_hidden_1/MatMul/ReadVariableOp2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp2H
"y0_hidden_2/BiasAdd/ReadVariableOp"y0_hidden_2/BiasAdd/ReadVariableOp2F
!y0_hidden_2/MatMul/ReadVariableOp!y0_hidden_2/MatMul/ReadVariableOp2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp2N
%y0_predictions/BiasAdd/ReadVariableOp%y0_predictions/BiasAdd/ReadVariableOp2L
$y0_predictions/MatMul/ReadVariableOp$y0_predictions/MatMul/ReadVariableOp2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp2H
"y1_hidden_1/BiasAdd/ReadVariableOp"y1_hidden_1/BiasAdd/ReadVariableOp2F
!y1_hidden_1/MatMul/ReadVariableOp!y1_hidden_1/MatMul/ReadVariableOp2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp2H
"y1_hidden_2/BiasAdd/ReadVariableOp"y1_hidden_2/BiasAdd/ReadVariableOp2F
!y1_hidden_2/MatMul/ReadVariableOp!y1_hidden_2/MatMul/ReadVariableOp2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp2N
%y1_predictions/BiasAdd/ReadVariableOp%y1_predictions/BiasAdd/ReadVariableOp2L
$y1_predictions/MatMul/ReadVariableOp$y1_predictions/MatMul/ReadVariableOp2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_phi_3_layer_call_fn_17871

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_3_layer_call_and_return_conditional_losses_16710p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_18082O
=y0_hidden_2_kernel_regularizer_square_readvariableop_resource:dd
identity��4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=y0_hidden_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&y0_hidden_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp
�
�
+__inference_y1_hidden_2_layer_call_fn_17969

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_16779o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_17934

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_y1_predictions_layer_call_and_return_conditional_losses_18036

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�7y1_predictions/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_17498	
input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:	�d
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:d

unknown_14:

unknown_15:d

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_16658o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�[
�
__inference__traced_save_18270
file_prefix+
'savev2_phi_1_kernel_read_readvariableop)
%savev2_phi_1_bias_read_readvariableop+
'savev2_phi_2_kernel_read_readvariableop)
%savev2_phi_2_bias_read_readvariableop+
'savev2_phi_3_kernel_read_readvariableop)
%savev2_phi_3_bias_read_readvariableop1
-savev2_y0_hidden_1_kernel_read_readvariableop/
+savev2_y0_hidden_1_bias_read_readvariableop1
-savev2_y1_hidden_1_kernel_read_readvariableop/
+savev2_y1_hidden_1_bias_read_readvariableop1
-savev2_y0_hidden_2_kernel_read_readvariableop/
+savev2_y0_hidden_2_bias_read_readvariableop1
-savev2_y1_hidden_2_kernel_read_readvariableop/
+savev2_y1_hidden_2_bias_read_readvariableop4
0savev2_y0_predictions_kernel_read_readvariableop2
.savev2_y0_predictions_bias_read_readvariableop4
0savev2_y1_predictions_kernel_read_readvariableop2
.savev2_y1_predictions_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_sgd_phi_1_kernel_momentum_read_readvariableop6
2savev2_sgd_phi_1_bias_momentum_read_readvariableop8
4savev2_sgd_phi_2_kernel_momentum_read_readvariableop6
2savev2_sgd_phi_2_bias_momentum_read_readvariableop8
4savev2_sgd_phi_3_kernel_momentum_read_readvariableop6
2savev2_sgd_phi_3_bias_momentum_read_readvariableop>
:savev2_sgd_y0_hidden_1_kernel_momentum_read_readvariableop<
8savev2_sgd_y0_hidden_1_bias_momentum_read_readvariableop>
:savev2_sgd_y1_hidden_1_kernel_momentum_read_readvariableop<
8savev2_sgd_y1_hidden_1_bias_momentum_read_readvariableop>
:savev2_sgd_y0_hidden_2_kernel_momentum_read_readvariableop<
8savev2_sgd_y0_hidden_2_bias_momentum_read_readvariableop>
:savev2_sgd_y1_hidden_2_kernel_momentum_read_readvariableop<
8savev2_sgd_y1_hidden_2_bias_momentum_read_readvariableopA
=savev2_sgd_y0_predictions_kernel_momentum_read_readvariableop?
;savev2_sgd_y0_predictions_bias_momentum_read_readvariableopA
=savev2_sgd_y1_predictions_kernel_momentum_read_readvariableop?
;savev2_sgd_y1_predictions_bias_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_phi_1_kernel_read_readvariableop%savev2_phi_1_bias_read_readvariableop'savev2_phi_2_kernel_read_readvariableop%savev2_phi_2_bias_read_readvariableop'savev2_phi_3_kernel_read_readvariableop%savev2_phi_3_bias_read_readvariableop-savev2_y0_hidden_1_kernel_read_readvariableop+savev2_y0_hidden_1_bias_read_readvariableop-savev2_y1_hidden_1_kernel_read_readvariableop+savev2_y1_hidden_1_bias_read_readvariableop-savev2_y0_hidden_2_kernel_read_readvariableop+savev2_y0_hidden_2_bias_read_readvariableop-savev2_y1_hidden_2_kernel_read_readvariableop+savev2_y1_hidden_2_bias_read_readvariableop0savev2_y0_predictions_kernel_read_readvariableop.savev2_y0_predictions_bias_read_readvariableop0savev2_y1_predictions_kernel_read_readvariableop.savev2_y1_predictions_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_sgd_phi_1_kernel_momentum_read_readvariableop2savev2_sgd_phi_1_bias_momentum_read_readvariableop4savev2_sgd_phi_2_kernel_momentum_read_readvariableop2savev2_sgd_phi_2_bias_momentum_read_readvariableop4savev2_sgd_phi_3_kernel_momentum_read_readvariableop2savev2_sgd_phi_3_bias_momentum_read_readvariableop:savev2_sgd_y0_hidden_1_kernel_momentum_read_readvariableop8savev2_sgd_y0_hidden_1_bias_momentum_read_readvariableop:savev2_sgd_y1_hidden_1_kernel_momentum_read_readvariableop8savev2_sgd_y1_hidden_1_bias_momentum_read_readvariableop:savev2_sgd_y0_hidden_2_kernel_momentum_read_readvariableop8savev2_sgd_y0_hidden_2_bias_momentum_read_readvariableop:savev2_sgd_y1_hidden_2_kernel_momentum_read_readvariableop8savev2_sgd_y1_hidden_2_bias_momentum_read_readvariableop=savev2_sgd_y0_predictions_kernel_momentum_read_readvariableop;savev2_sgd_y0_predictions_bias_momentum_read_readvariableop=savev2_sgd_y1_predictions_kernel_momentum_read_readvariableop;savev2_sgd_y1_predictions_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:�:
��:�:
��:�:	�d:d:	�d:d:dd:d:dd:d:d::d:: : : : : : : : :	�:�:
��:�:
��:�:	�d:d:	�d:d:dd:d:dd:d:d::d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d:%	!

_output_shapes
:	�d: 


_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:! 

_output_shapes	
:�:%!!

_output_shapes
:	�d: "

_output_shapes
:d:%#!

_output_shapes
:	�d: $

_output_shapes
:d:$% 

_output_shapes

:dd: &

_output_shapes
:d:$' 

_output_shapes

:dd: (

_output_shapes
:d:$) 

_output_shapes

:d: *

_output_shapes
::$+ 

_output_shapes

:d: ,

_output_shapes
::-

_output_shapes
: 
�
�
%__inference_model_layer_call_fn_17575

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:	�d
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:d

unknown_14:

unknown_15:d

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_16898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
@__inference_phi_2_layer_call_and_return_conditional_losses_17862

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_18093O
=y1_hidden_2_kernel_regularizer_square_readvariableop_resource:dd
identity��4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=y1_hidden_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&y1_hidden_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp
�
r
F__inference_concatenate_layer_call_and_return_conditional_losses_18049
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�e
�

@__inference_model_layer_call_and_return_conditional_losses_16898

inputs
phi_1_16677:	�
phi_1_16679:	�
phi_2_16694:
��
phi_2_16696:	�
phi_3_16711:
��
phi_3_16713:	�$
y1_hidden_1_16734:	�d
y1_hidden_1_16736:d$
y0_hidden_1_16757:	�d
y0_hidden_1_16759:d#
y1_hidden_2_16780:dd
y1_hidden_2_16782:d#
y0_hidden_2_16803:dd
y0_hidden_2_16805:d&
y0_predictions_16825:d"
y0_predictions_16827:&
y1_predictions_16847:d"
y1_predictions_16849:
identity��phi_1/StatefulPartitionedCall�phi_2/StatefulPartitionedCall�phi_3/StatefulPartitionedCall�#y0_hidden_1/StatefulPartitionedCall�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y0_hidden_2/StatefulPartitionedCall�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y0_predictions/StatefulPartitionedCall�7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_1/StatefulPartitionedCall�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_2/StatefulPartitionedCall�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y1_predictions/StatefulPartitionedCall�7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
phi_1/StatefulPartitionedCallStatefulPartitionedCallinputsphi_1_16677phi_1_16679*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_1_layer_call_and_return_conditional_losses_16676�
phi_2/StatefulPartitionedCallStatefulPartitionedCall&phi_1/StatefulPartitionedCall:output:0phi_2_16694phi_2_16696*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_2_layer_call_and_return_conditional_losses_16693�
phi_3/StatefulPartitionedCallStatefulPartitionedCall&phi_2/StatefulPartitionedCall:output:0phi_3_16711phi_3_16713*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_3_layer_call_and_return_conditional_losses_16710�
#y1_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y1_hidden_1_16734y1_hidden_1_16736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_16733�
#y0_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y0_hidden_1_16757y0_hidden_1_16759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_16756�
#y1_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_1/StatefulPartitionedCall:output:0y1_hidden_2_16780y1_hidden_2_16782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_16779�
#y0_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_1/StatefulPartitionedCall:output:0y0_hidden_2_16803y0_hidden_2_16805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_16802�
&y0_predictions/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_2/StatefulPartitionedCall:output:0y0_predictions_16825y0_predictions_16827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y0_predictions_layer_call_and_return_conditional_losses_16824�
&y1_predictions/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_2/StatefulPartitionedCall:output:0y1_predictions_16847y1_predictions_16849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y1_predictions_layer_call_and_return_conditional_losses_16846�
concatenate/PartitionedCallPartitionedCall/y0_predictions/StatefulPartitionedCall:output:0/y1_predictions/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16859�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_1_16757*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_1_16734*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_2_16803*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_2_16780*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_predictions_16825*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_predictions_16847*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^phi_1/StatefulPartitionedCall^phi_2/StatefulPartitionedCall^phi_3/StatefulPartitionedCall$^y0_hidden_1/StatefulPartitionedCall5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y0_hidden_2/StatefulPartitionedCall5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y0_predictions/StatefulPartitionedCall8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_1/StatefulPartitionedCall5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_2/StatefulPartitionedCall5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y1_predictions/StatefulPartitionedCall8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2>
phi_1/StatefulPartitionedCallphi_1/StatefulPartitionedCall2>
phi_2/StatefulPartitionedCallphi_2/StatefulPartitionedCall2>
phi_3/StatefulPartitionedCallphi_3/StatefulPartitionedCall2J
#y0_hidden_1/StatefulPartitionedCall#y0_hidden_1/StatefulPartitionedCall2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y0_hidden_2/StatefulPartitionedCall#y0_hidden_2/StatefulPartitionedCall2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y0_predictions/StatefulPartitionedCall&y0_predictions/StatefulPartitionedCall2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_1/StatefulPartitionedCall#y1_hidden_1/StatefulPartitionedCall2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_2/StatefulPartitionedCall#y1_hidden_2/StatefulPartitionedCall2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y1_predictions/StatefulPartitionedCall&y1_predictions/StatefulPartitionedCall2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_18412
file_prefix0
assignvariableop_phi_1_kernel:	�,
assignvariableop_1_phi_1_bias:	�3
assignvariableop_2_phi_2_kernel:
��,
assignvariableop_3_phi_2_bias:	�3
assignvariableop_4_phi_3_kernel:
��,
assignvariableop_5_phi_3_bias:	�8
%assignvariableop_6_y0_hidden_1_kernel:	�d1
#assignvariableop_7_y0_hidden_1_bias:d8
%assignvariableop_8_y1_hidden_1_kernel:	�d1
#assignvariableop_9_y1_hidden_1_bias:d8
&assignvariableop_10_y0_hidden_2_kernel:dd2
$assignvariableop_11_y0_hidden_2_bias:d8
&assignvariableop_12_y1_hidden_2_kernel:dd2
$assignvariableop_13_y1_hidden_2_bias:d;
)assignvariableop_14_y0_predictions_kernel:d5
'assignvariableop_15_y0_predictions_bias:;
)assignvariableop_16_y1_predictions_kernel:d5
'assignvariableop_17_y1_predictions_bias:&
assignvariableop_18_sgd_iter:	 '
assignvariableop_19_sgd_decay: /
%assignvariableop_20_sgd_learning_rate: *
 assignvariableop_21_sgd_momentum: %
assignvariableop_22_total_1: %
assignvariableop_23_count_1: #
assignvariableop_24_total: #
assignvariableop_25_count: @
-assignvariableop_26_sgd_phi_1_kernel_momentum:	�:
+assignvariableop_27_sgd_phi_1_bias_momentum:	�A
-assignvariableop_28_sgd_phi_2_kernel_momentum:
��:
+assignvariableop_29_sgd_phi_2_bias_momentum:	�A
-assignvariableop_30_sgd_phi_3_kernel_momentum:
��:
+assignvariableop_31_sgd_phi_3_bias_momentum:	�F
3assignvariableop_32_sgd_y0_hidden_1_kernel_momentum:	�d?
1assignvariableop_33_sgd_y0_hidden_1_bias_momentum:dF
3assignvariableop_34_sgd_y1_hidden_1_kernel_momentum:	�d?
1assignvariableop_35_sgd_y1_hidden_1_bias_momentum:dE
3assignvariableop_36_sgd_y0_hidden_2_kernel_momentum:dd?
1assignvariableop_37_sgd_y0_hidden_2_bias_momentum:dE
3assignvariableop_38_sgd_y1_hidden_2_kernel_momentum:dd?
1assignvariableop_39_sgd_y1_hidden_2_bias_momentum:dH
6assignvariableop_40_sgd_y0_predictions_kernel_momentum:dB
4assignvariableop_41_sgd_y0_predictions_bias_momentum:H
6assignvariableop_42_sgd_y1_predictions_kernel_momentum:dB
4assignvariableop_43_sgd_y1_predictions_bias_momentum:
identity_45��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_phi_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_phi_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_phi_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_phi_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_phi_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_phi_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_y0_hidden_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_y0_hidden_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_y1_hidden_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_y1_hidden_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_y0_hidden_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_y0_hidden_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_y1_hidden_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_y1_hidden_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_y0_predictions_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_y0_predictions_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_y1_predictions_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_y1_predictions_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_sgd_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_sgd_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_sgd_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_sgd_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp-assignvariableop_26_sgd_phi_1_kernel_momentumIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_sgd_phi_1_bias_momentumIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp-assignvariableop_28_sgd_phi_2_kernel_momentumIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_sgd_phi_2_bias_momentumIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_sgd_phi_3_kernel_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_sgd_phi_3_bias_momentumIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp3assignvariableop_32_sgd_y0_hidden_1_kernel_momentumIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp1assignvariableop_33_sgd_y0_hidden_1_bias_momentumIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_sgd_y1_hidden_1_kernel_momentumIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp1assignvariableop_35_sgd_y1_hidden_1_bias_momentumIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp3assignvariableop_36_sgd_y0_hidden_2_kernel_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp1assignvariableop_37_sgd_y0_hidden_2_bias_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp3assignvariableop_38_sgd_y1_hidden_2_kernel_momentumIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp1assignvariableop_39_sgd_y1_hidden_2_bias_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp6assignvariableop_40_sgd_y0_predictions_kernel_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp4assignvariableop_41_sgd_y0_predictions_bias_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_sgd_y1_predictions_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp4assignvariableop_43_sgd_y1_predictions_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_45IdentityIdentity_44:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
@__inference_phi_1_layer_call_and_return_conditional_losses_16676

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
@__inference_phi_3_layer_call_and_return_conditional_losses_16710

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_y0_predictions_layer_call_and_return_conditional_losses_18011

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�7y0_predictions/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_17986

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
p
F__inference_concatenate_layer_call_and_return_conditional_losses_16859

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_17822

inputs7
$phi_1_matmul_readvariableop_resource:	�4
%phi_1_biasadd_readvariableop_resource:	�8
$phi_2_matmul_readvariableop_resource:
��4
%phi_2_biasadd_readvariableop_resource:	�8
$phi_3_matmul_readvariableop_resource:
��4
%phi_3_biasadd_readvariableop_resource:	�=
*y1_hidden_1_matmul_readvariableop_resource:	�d9
+y1_hidden_1_biasadd_readvariableop_resource:d=
*y0_hidden_1_matmul_readvariableop_resource:	�d9
+y0_hidden_1_biasadd_readvariableop_resource:d<
*y1_hidden_2_matmul_readvariableop_resource:dd9
+y1_hidden_2_biasadd_readvariableop_resource:d<
*y0_hidden_2_matmul_readvariableop_resource:dd9
+y0_hidden_2_biasadd_readvariableop_resource:d?
-y0_predictions_matmul_readvariableop_resource:d<
.y0_predictions_biasadd_readvariableop_resource:?
-y1_predictions_matmul_readvariableop_resource:d<
.y1_predictions_biasadd_readvariableop_resource:
identity��phi_1/BiasAdd/ReadVariableOp�phi_1/MatMul/ReadVariableOp�phi_2/BiasAdd/ReadVariableOp�phi_2/MatMul/ReadVariableOp�phi_3/BiasAdd/ReadVariableOp�phi_3/MatMul/ReadVariableOp�"y0_hidden_1/BiasAdd/ReadVariableOp�!y0_hidden_1/MatMul/ReadVariableOp�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�"y0_hidden_2/BiasAdd/ReadVariableOp�!y0_hidden_2/MatMul/ReadVariableOp�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�%y0_predictions/BiasAdd/ReadVariableOp�$y0_predictions/MatMul/ReadVariableOp�7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�"y1_hidden_1/BiasAdd/ReadVariableOp�!y1_hidden_1/MatMul/ReadVariableOp�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�"y1_hidden_2/BiasAdd/ReadVariableOp�!y1_hidden_2/MatMul/ReadVariableOp�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�%y1_predictions/BiasAdd/ReadVariableOp�$y1_predictions/MatMul/ReadVariableOp�7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
phi_1/MatMul/ReadVariableOpReadVariableOp$phi_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0v
phi_1/MatMulMatMulinputs#phi_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
phi_1/BiasAdd/ReadVariableOpReadVariableOp%phi_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
phi_1/BiasAddBiasAddphi_1/MatMul:product:0$phi_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
	phi_1/EluEluphi_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
phi_2/MatMul/ReadVariableOpReadVariableOp$phi_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
phi_2/MatMulMatMulphi_1/Elu:activations:0#phi_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
phi_2/BiasAdd/ReadVariableOpReadVariableOp%phi_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
phi_2/BiasAddBiasAddphi_2/MatMul:product:0$phi_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
	phi_2/EluEluphi_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
phi_3/MatMul/ReadVariableOpReadVariableOp$phi_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
phi_3/MatMulMatMulphi_2/Elu:activations:0#phi_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
phi_3/BiasAdd/ReadVariableOpReadVariableOp%phi_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
phi_3/BiasAddBiasAddphi_3/MatMul:product:0$phi_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������[
	phi_3/EluEluphi_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!y1_hidden_1/MatMul/ReadVariableOpReadVariableOp*y1_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
y1_hidden_1/MatMulMatMulphi_3/Elu:activations:0)y1_hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y1_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+y1_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y1_hidden_1/BiasAddBiasAddy1_hidden_1/MatMul:product:0*y1_hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y1_hidden_1/EluEluy1_hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
!y0_hidden_1/MatMul/ReadVariableOpReadVariableOp*y0_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
y0_hidden_1/MatMulMatMulphi_3/Elu:activations:0)y0_hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y0_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+y0_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y0_hidden_1/BiasAddBiasAddy0_hidden_1/MatMul:product:0*y0_hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y0_hidden_1/EluEluy0_hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
!y1_hidden_2/MatMul/ReadVariableOpReadVariableOp*y1_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
y1_hidden_2/MatMulMatMuly1_hidden_1/Elu:activations:0)y1_hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y1_hidden_2/BiasAdd/ReadVariableOpReadVariableOp+y1_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y1_hidden_2/BiasAddBiasAddy1_hidden_2/MatMul:product:0*y1_hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y1_hidden_2/EluEluy1_hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
!y0_hidden_2/MatMul/ReadVariableOpReadVariableOp*y0_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
y0_hidden_2/MatMulMatMuly0_hidden_1/Elu:activations:0)y0_hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"y0_hidden_2/BiasAdd/ReadVariableOpReadVariableOp+y0_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
y0_hidden_2/BiasAddBiasAddy0_hidden_2/MatMul:product:0*y0_hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
y0_hidden_2/EluEluy0_hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
$y0_predictions/MatMul/ReadVariableOpReadVariableOp-y0_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
y0_predictions/MatMulMatMuly0_hidden_2/Elu:activations:0,y0_predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%y0_predictions/BiasAdd/ReadVariableOpReadVariableOp.y0_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
y0_predictions/BiasAddBiasAddy0_predictions/MatMul:product:0-y0_predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$y1_predictions/MatMul/ReadVariableOpReadVariableOp-y1_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
y1_predictions/MatMulMatMuly1_hidden_2/Elu:activations:0,y1_predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%y1_predictions/BiasAdd/ReadVariableOpReadVariableOp.y1_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
y1_predictions/BiasAddBiasAddy1_predictions/MatMul:product:0-y1_predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2y0_predictions/BiasAdd:output:0y1_predictions/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y0_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y1_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y0_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*y1_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-y0_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-y1_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityconcatenate/concat:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^phi_1/BiasAdd/ReadVariableOp^phi_1/MatMul/ReadVariableOp^phi_2/BiasAdd/ReadVariableOp^phi_2/MatMul/ReadVariableOp^phi_3/BiasAdd/ReadVariableOp^phi_3/MatMul/ReadVariableOp#^y0_hidden_1/BiasAdd/ReadVariableOp"^y0_hidden_1/MatMul/ReadVariableOp5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp#^y0_hidden_2/BiasAdd/ReadVariableOp"^y0_hidden_2/MatMul/ReadVariableOp5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp&^y0_predictions/BiasAdd/ReadVariableOp%^y0_predictions/MatMul/ReadVariableOp8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp#^y1_hidden_1/BiasAdd/ReadVariableOp"^y1_hidden_1/MatMul/ReadVariableOp5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp#^y1_hidden_2/BiasAdd/ReadVariableOp"^y1_hidden_2/MatMul/ReadVariableOp5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp&^y1_predictions/BiasAdd/ReadVariableOp%^y1_predictions/MatMul/ReadVariableOp8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2<
phi_1/BiasAdd/ReadVariableOpphi_1/BiasAdd/ReadVariableOp2:
phi_1/MatMul/ReadVariableOpphi_1/MatMul/ReadVariableOp2<
phi_2/BiasAdd/ReadVariableOpphi_2/BiasAdd/ReadVariableOp2:
phi_2/MatMul/ReadVariableOpphi_2/MatMul/ReadVariableOp2<
phi_3/BiasAdd/ReadVariableOpphi_3/BiasAdd/ReadVariableOp2:
phi_3/MatMul/ReadVariableOpphi_3/MatMul/ReadVariableOp2H
"y0_hidden_1/BiasAdd/ReadVariableOp"y0_hidden_1/BiasAdd/ReadVariableOp2F
!y0_hidden_1/MatMul/ReadVariableOp!y0_hidden_1/MatMul/ReadVariableOp2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp2H
"y0_hidden_2/BiasAdd/ReadVariableOp"y0_hidden_2/BiasAdd/ReadVariableOp2F
!y0_hidden_2/MatMul/ReadVariableOp!y0_hidden_2/MatMul/ReadVariableOp2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp2N
%y0_predictions/BiasAdd/ReadVariableOp%y0_predictions/BiasAdd/ReadVariableOp2L
$y0_predictions/MatMul/ReadVariableOp$y0_predictions/MatMul/ReadVariableOp2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp2H
"y1_hidden_1/BiasAdd/ReadVariableOp"y1_hidden_1/BiasAdd/ReadVariableOp2F
!y1_hidden_1/MatMul/ReadVariableOp!y1_hidden_1/MatMul/ReadVariableOp2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp2H
"y1_hidden_2/BiasAdd/ReadVariableOp"y1_hidden_2/BiasAdd/ReadVariableOp2F
!y1_hidden_2/MatMul/ReadVariableOp!y1_hidden_2/MatMul/ReadVariableOp2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp2N
%y1_predictions/BiasAdd/ReadVariableOp%y1_predictions/BiasAdd/ReadVariableOp2L
$y1_predictions/MatMul/ReadVariableOp$y1_predictions/MatMul/ReadVariableOp2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_y0_hidden_1_layer_call_fn_17891

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_16756o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
W
+__inference_concatenate_layer_call_fn_18042
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16859`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
__inference_loss_fn_1_18071P
=y1_hidden_1_kernel_regularizer_square_readvariableop_resource:	�d
identity��4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=y1_hidden_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&y1_hidden_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp
�
�
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_17960

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
@__inference_phi_1_layer_call_and_return_conditional_losses_17842

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_18115R
@y1_predictions_kernel_regularizer_square_readvariableop_resource:d
identity��7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@y1_predictions_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentity)y1_predictions/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp
�

�
@__inference_phi_3_layer_call_and_return_conditional_losses_17882

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_17908

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_y0_predictions_layer_call_fn_17995

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y0_predictions_layer_call_and_return_conditional_losses_16824o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
.__inference_y1_predictions_layer_call_fn_18020

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y1_predictions_layer_call_and_return_conditional_losses_16846o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_17616

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:	�d
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:d

unknown_14:

unknown_15:d

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_17163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Y
�
 __inference__wrapped_model_16658	
input=
*model_phi_1_matmul_readvariableop_resource:	�:
+model_phi_1_biasadd_readvariableop_resource:	�>
*model_phi_2_matmul_readvariableop_resource:
��:
+model_phi_2_biasadd_readvariableop_resource:	�>
*model_phi_3_matmul_readvariableop_resource:
��:
+model_phi_3_biasadd_readvariableop_resource:	�C
0model_y1_hidden_1_matmul_readvariableop_resource:	�d?
1model_y1_hidden_1_biasadd_readvariableop_resource:dC
0model_y0_hidden_1_matmul_readvariableop_resource:	�d?
1model_y0_hidden_1_biasadd_readvariableop_resource:dB
0model_y1_hidden_2_matmul_readvariableop_resource:dd?
1model_y1_hidden_2_biasadd_readvariableop_resource:dB
0model_y0_hidden_2_matmul_readvariableop_resource:dd?
1model_y0_hidden_2_biasadd_readvariableop_resource:dE
3model_y0_predictions_matmul_readvariableop_resource:dB
4model_y0_predictions_biasadd_readvariableop_resource:E
3model_y1_predictions_matmul_readvariableop_resource:dB
4model_y1_predictions_biasadd_readvariableop_resource:
identity��"model/phi_1/BiasAdd/ReadVariableOp�!model/phi_1/MatMul/ReadVariableOp�"model/phi_2/BiasAdd/ReadVariableOp�!model/phi_2/MatMul/ReadVariableOp�"model/phi_3/BiasAdd/ReadVariableOp�!model/phi_3/MatMul/ReadVariableOp�(model/y0_hidden_1/BiasAdd/ReadVariableOp�'model/y0_hidden_1/MatMul/ReadVariableOp�(model/y0_hidden_2/BiasAdd/ReadVariableOp�'model/y0_hidden_2/MatMul/ReadVariableOp�+model/y0_predictions/BiasAdd/ReadVariableOp�*model/y0_predictions/MatMul/ReadVariableOp�(model/y1_hidden_1/BiasAdd/ReadVariableOp�'model/y1_hidden_1/MatMul/ReadVariableOp�(model/y1_hidden_2/BiasAdd/ReadVariableOp�'model/y1_hidden_2/MatMul/ReadVariableOp�+model/y1_predictions/BiasAdd/ReadVariableOp�*model/y1_predictions/MatMul/ReadVariableOp�
!model/phi_1/MatMul/ReadVariableOpReadVariableOp*model_phi_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/phi_1/MatMulMatMulinput)model/phi_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/phi_1/BiasAdd/ReadVariableOpReadVariableOp+model_phi_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/phi_1/BiasAddBiasAddmodel/phi_1/MatMul:product:0*model/phi_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
model/phi_1/EluElumodel/phi_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!model/phi_2/MatMul/ReadVariableOpReadVariableOp*model_phi_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/phi_2/MatMulMatMulmodel/phi_1/Elu:activations:0)model/phi_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/phi_2/BiasAdd/ReadVariableOpReadVariableOp+model_phi_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/phi_2/BiasAddBiasAddmodel/phi_2/MatMul:product:0*model/phi_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
model/phi_2/EluElumodel/phi_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!model/phi_3/MatMul/ReadVariableOpReadVariableOp*model_phi_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/phi_3/MatMulMatMulmodel/phi_2/Elu:activations:0)model/phi_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/phi_3/BiasAdd/ReadVariableOpReadVariableOp+model_phi_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/phi_3/BiasAddBiasAddmodel/phi_3/MatMul:product:0*model/phi_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
model/phi_3/EluElumodel/phi_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model/y1_hidden_1/MatMul/ReadVariableOpReadVariableOp0model_y1_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
model/y1_hidden_1/MatMulMatMulmodel/phi_3/Elu:activations:0/model/y1_hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(model/y1_hidden_1/BiasAdd/ReadVariableOpReadVariableOp1model_y1_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/y1_hidden_1/BiasAddBiasAdd"model/y1_hidden_1/MatMul:product:00model/y1_hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
model/y1_hidden_1/EluElu"model/y1_hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
'model/y0_hidden_1/MatMul/ReadVariableOpReadVariableOp0model_y0_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
model/y0_hidden_1/MatMulMatMulmodel/phi_3/Elu:activations:0/model/y0_hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(model/y0_hidden_1/BiasAdd/ReadVariableOpReadVariableOp1model_y0_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/y0_hidden_1/BiasAddBiasAdd"model/y0_hidden_1/MatMul:product:00model/y0_hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
model/y0_hidden_1/EluElu"model/y0_hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
'model/y1_hidden_2/MatMul/ReadVariableOpReadVariableOp0model_y1_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
model/y1_hidden_2/MatMulMatMul#model/y1_hidden_1/Elu:activations:0/model/y1_hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(model/y1_hidden_2/BiasAdd/ReadVariableOpReadVariableOp1model_y1_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/y1_hidden_2/BiasAddBiasAdd"model/y1_hidden_2/MatMul:product:00model/y1_hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
model/y1_hidden_2/EluElu"model/y1_hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
'model/y0_hidden_2/MatMul/ReadVariableOpReadVariableOp0model_y0_hidden_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
model/y0_hidden_2/MatMulMatMul#model/y0_hidden_1/Elu:activations:0/model/y0_hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(model/y0_hidden_2/BiasAdd/ReadVariableOpReadVariableOp1model_y0_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/y0_hidden_2/BiasAddBiasAdd"model/y0_hidden_2/MatMul:product:00model/y0_hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
model/y0_hidden_2/EluElu"model/y0_hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
*model/y0_predictions/MatMul/ReadVariableOpReadVariableOp3model_y0_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
model/y0_predictions/MatMulMatMul#model/y0_hidden_2/Elu:activations:02model/y0_predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+model/y0_predictions/BiasAdd/ReadVariableOpReadVariableOp4model_y0_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/y0_predictions/BiasAddBiasAdd%model/y0_predictions/MatMul:product:03model/y0_predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model/y1_predictions/MatMul/ReadVariableOpReadVariableOp3model_y1_predictions_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
model/y1_predictions/MatMulMatMul#model/y1_hidden_2/Elu:activations:02model/y1_predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+model/y1_predictions/BiasAdd/ReadVariableOpReadVariableOp4model_y1_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/y1_predictions/BiasAddBiasAdd%model/y1_predictions/MatMul:product:03model/y1_predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2%model/y0_predictions/BiasAdd:output:0%model/y1_predictions/BiasAdd:output:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������p
IdentityIdentity!model/concatenate/concat:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/phi_1/BiasAdd/ReadVariableOp"^model/phi_1/MatMul/ReadVariableOp#^model/phi_2/BiasAdd/ReadVariableOp"^model/phi_2/MatMul/ReadVariableOp#^model/phi_3/BiasAdd/ReadVariableOp"^model/phi_3/MatMul/ReadVariableOp)^model/y0_hidden_1/BiasAdd/ReadVariableOp(^model/y0_hidden_1/MatMul/ReadVariableOp)^model/y0_hidden_2/BiasAdd/ReadVariableOp(^model/y0_hidden_2/MatMul/ReadVariableOp,^model/y0_predictions/BiasAdd/ReadVariableOp+^model/y0_predictions/MatMul/ReadVariableOp)^model/y1_hidden_1/BiasAdd/ReadVariableOp(^model/y1_hidden_1/MatMul/ReadVariableOp)^model/y1_hidden_2/BiasAdd/ReadVariableOp(^model/y1_hidden_2/MatMul/ReadVariableOp,^model/y1_predictions/BiasAdd/ReadVariableOp+^model/y1_predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2H
"model/phi_1/BiasAdd/ReadVariableOp"model/phi_1/BiasAdd/ReadVariableOp2F
!model/phi_1/MatMul/ReadVariableOp!model/phi_1/MatMul/ReadVariableOp2H
"model/phi_2/BiasAdd/ReadVariableOp"model/phi_2/BiasAdd/ReadVariableOp2F
!model/phi_2/MatMul/ReadVariableOp!model/phi_2/MatMul/ReadVariableOp2H
"model/phi_3/BiasAdd/ReadVariableOp"model/phi_3/BiasAdd/ReadVariableOp2F
!model/phi_3/MatMul/ReadVariableOp!model/phi_3/MatMul/ReadVariableOp2T
(model/y0_hidden_1/BiasAdd/ReadVariableOp(model/y0_hidden_1/BiasAdd/ReadVariableOp2R
'model/y0_hidden_1/MatMul/ReadVariableOp'model/y0_hidden_1/MatMul/ReadVariableOp2T
(model/y0_hidden_2/BiasAdd/ReadVariableOp(model/y0_hidden_2/BiasAdd/ReadVariableOp2R
'model/y0_hidden_2/MatMul/ReadVariableOp'model/y0_hidden_2/MatMul/ReadVariableOp2Z
+model/y0_predictions/BiasAdd/ReadVariableOp+model/y0_predictions/BiasAdd/ReadVariableOp2X
*model/y0_predictions/MatMul/ReadVariableOp*model/y0_predictions/MatMul/ReadVariableOp2T
(model/y1_hidden_1/BiasAdd/ReadVariableOp(model/y1_hidden_1/BiasAdd/ReadVariableOp2R
'model/y1_hidden_1/MatMul/ReadVariableOp'model/y1_hidden_1/MatMul/ReadVariableOp2T
(model/y1_hidden_2/BiasAdd/ReadVariableOp(model/y1_hidden_2/BiasAdd/ReadVariableOp2R
'model/y1_hidden_2/MatMul/ReadVariableOp'model/y1_hidden_2/MatMul/ReadVariableOp2Z
+model/y1_predictions/BiasAdd/ReadVariableOp+model/y1_predictions/BiasAdd/ReadVariableOp2X
*model/y1_predictions/MatMul/ReadVariableOp*model/y1_predictions/MatMul/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameinput
�e
�

@__inference_model_layer_call_and_return_conditional_losses_17415	
input
phi_1_17332:	�
phi_1_17334:	�
phi_2_17337:
��
phi_2_17339:	�
phi_3_17342:
��
phi_3_17344:	�$
y1_hidden_1_17347:	�d
y1_hidden_1_17349:d$
y0_hidden_1_17352:	�d
y0_hidden_1_17354:d#
y1_hidden_2_17357:dd
y1_hidden_2_17359:d#
y0_hidden_2_17362:dd
y0_hidden_2_17364:d&
y0_predictions_17367:d"
y0_predictions_17369:&
y1_predictions_17372:d"
y1_predictions_17374:
identity��phi_1/StatefulPartitionedCall�phi_2/StatefulPartitionedCall�phi_3/StatefulPartitionedCall�#y0_hidden_1/StatefulPartitionedCall�4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y0_hidden_2/StatefulPartitionedCall�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y0_predictions/StatefulPartitionedCall�7y0_predictions/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_1/StatefulPartitionedCall�4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp�#y1_hidden_2/StatefulPartitionedCall�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp�&y1_predictions/StatefulPartitionedCall�7y1_predictions/kernel/Regularizer/Square/ReadVariableOp�
phi_1/StatefulPartitionedCallStatefulPartitionedCallinputphi_1_17332phi_1_17334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_1_layer_call_and_return_conditional_losses_16676�
phi_2/StatefulPartitionedCallStatefulPartitionedCall&phi_1/StatefulPartitionedCall:output:0phi_2_17337phi_2_17339*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_2_layer_call_and_return_conditional_losses_16693�
phi_3/StatefulPartitionedCallStatefulPartitionedCall&phi_2/StatefulPartitionedCall:output:0phi_3_17342phi_3_17344*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_phi_3_layer_call_and_return_conditional_losses_16710�
#y1_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y1_hidden_1_17347y1_hidden_1_17349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_16733�
#y0_hidden_1/StatefulPartitionedCallStatefulPartitionedCall&phi_3/StatefulPartitionedCall:output:0y0_hidden_1_17352y0_hidden_1_17354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_16756�
#y1_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_1/StatefulPartitionedCall:output:0y1_hidden_2_17357y1_hidden_2_17359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_16779�
#y0_hidden_2/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_1/StatefulPartitionedCall:output:0y0_hidden_2_17362y0_hidden_2_17364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_16802�
&y0_predictions/StatefulPartitionedCallStatefulPartitionedCall,y0_hidden_2/StatefulPartitionedCall:output:0y0_predictions_17367y0_predictions_17369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y0_predictions_layer_call_and_return_conditional_losses_16824�
&y1_predictions/StatefulPartitionedCallStatefulPartitionedCall,y1_hidden_2/StatefulPartitionedCall:output:0y1_predictions_17372y1_predictions_17374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_y1_predictions_layer_call_and_return_conditional_losses_16846�
concatenate/PartitionedCallPartitionedCall/y0_predictions/StatefulPartitionedCall:output:0/y1_predictions/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16859�
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_1_17352*
_output_shapes
:	�d*
dtype0�
%y0_hidden_1/kernel/Regularizer/SquareSquare<y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y0_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_1/kernel/Regularizer/SumSum)y0_hidden_1/kernel/Regularizer/Square:y:0-y0_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_1/kernel/Regularizer/mulMul-y0_hidden_1/kernel/Regularizer/mul/x:output:0+y0_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_1_17347*
_output_shapes
:	�d*
dtype0�
%y1_hidden_1/kernel/Regularizer/SquareSquare<y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�du
$y1_hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_1/kernel/Regularizer/SumSum)y1_hidden_1/kernel/Regularizer/Square:y:0-y1_hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_1/kernel/Regularizer/mulMul-y1_hidden_1/kernel/Regularizer/mul/x:output:0+y1_hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_hidden_2_17362*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_hidden_2_17357*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y0_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy0_predictions_17367*
_output_shapes

:d*
dtype0�
(y0_predictions/kernel/Regularizer/SquareSquare?y0_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y0_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y0_predictions/kernel/Regularizer/SumSum,y0_predictions/kernel/Regularizer/Square:y:00y0_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y0_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y0_predictions/kernel/Regularizer/mulMul0y0_predictions/kernel/Regularizer/mul/x:output:0.y0_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
7y1_predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpy1_predictions_17372*
_output_shapes

:d*
dtype0�
(y1_predictions/kernel/Regularizer/SquareSquare?y1_predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dx
'y1_predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%y1_predictions/kernel/Regularizer/SumSum,y1_predictions/kernel/Regularizer/Square:y:00y1_predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'y1_predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
%y1_predictions/kernel/Regularizer/mulMul0y1_predictions/kernel/Regularizer/mul/x:output:0.y1_predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^phi_1/StatefulPartitionedCall^phi_2/StatefulPartitionedCall^phi_3/StatefulPartitionedCall$^y0_hidden_1/StatefulPartitionedCall5^y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y0_hidden_2/StatefulPartitionedCall5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y0_predictions/StatefulPartitionedCall8^y0_predictions/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_1/StatefulPartitionedCall5^y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp$^y1_hidden_2/StatefulPartitionedCall5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp'^y1_predictions/StatefulPartitionedCall8^y1_predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2>
phi_1/StatefulPartitionedCallphi_1/StatefulPartitionedCall2>
phi_2/StatefulPartitionedCallphi_2/StatefulPartitionedCall2>
phi_3/StatefulPartitionedCallphi_3/StatefulPartitionedCall2J
#y0_hidden_1/StatefulPartitionedCall#y0_hidden_1/StatefulPartitionedCall2l
4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y0_hidden_2/StatefulPartitionedCall#y0_hidden_2/StatefulPartitionedCall2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y0_predictions/StatefulPartitionedCall&y0_predictions/StatefulPartitionedCall2r
7y0_predictions/kernel/Regularizer/Square/ReadVariableOp7y0_predictions/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_1/StatefulPartitionedCall#y1_hidden_1/StatefulPartitionedCall2l
4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_1/kernel/Regularizer/Square/ReadVariableOp2J
#y1_hidden_2/StatefulPartitionedCall#y1_hidden_2/StatefulPartitionedCall2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp2P
&y1_predictions/StatefulPartitionedCall&y1_predictions/StatefulPartitionedCall2r
7y1_predictions/kernel/Regularizer/Square/ReadVariableOp7y1_predictions/kernel/Regularizer/Square/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_16802

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y0_hidden_2/kernel/Regularizer/SquareSquare<y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y0_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y0_hidden_2/kernel/Regularizer/SumSum)y0_hidden_2/kernel/Regularizer/Square:y:0-y0_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y0_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y0_hidden_2/kernel/Regularizer/mulMul-y0_hidden_2/kernel/Regularizer/mul/x:output:0+y0_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y0_hidden_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_17243	
input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:	�d
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:d

unknown_14:

unknown_15:d

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_17163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_16779

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%y1_hidden_2/kernel/Regularizer/SquareSquare<y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ddu
$y1_hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
"y1_hidden_2/kernel/Regularizer/SumSum)y1_hidden_2/kernel/Regularizer/Square:y:0-y1_hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$y1_hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"y1_hidden_2/kernel/Regularizer/mulMul-y1_hidden_2/kernel/Regularizer/mul/x:output:0+y1_hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp4y1_hidden_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
input.
serving_default_input:0���������?
concatenate0
StatefulPartitionedCall:0���������tensorflow/serving/predict:ށ
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17"
trackable_list_wrapper
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17"
trackable_list_wrapper
J
c0
d1
e2
f3
g4
h5"
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_0
otrace_1
ptrace_2
qtrace_32�
%__inference_model_layer_call_fn_16937
%__inference_model_layer_call_fn_17575
%__inference_model_layer_call_fn_17616
%__inference_model_layer_call_fn_17243�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zntrace_0zotrace_1zptrace_2zqtrace_3
�
rtrace_0
strace_1
ttrace_2
utrace_32�
@__inference_model_layer_call_and_return_conditional_losses_17719
@__inference_model_layer_call_and_return_conditional_losses_17822
@__inference_model_layer_call_and_return_conditional_losses_17329
@__inference_model_layer_call_and_return_conditional_losses_17415�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zrtrace_0zstrace_1zttrace_2zutrace_3
�B�
 __inference__wrapped_model_16658input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
viter
	wdecay
xlearning_rate
ymomentummomentum�momentum�#momentum�$momentum�+momentum�,momentum�3momentum�4momentum�;momentum�<momentum�Cmomentum�Dmomentum�Kmomentum�Lmomentum�Smomentum�Tmomentum�[momentum�\momentum�"
	optimizer
,
zserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_phi_1_layer_call_fn_17831�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_phi_1_layer_call_and_return_conditional_losses_17842�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2phi_1/kernel
:�2
phi_1/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_phi_2_layer_call_fn_17851�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_phi_2_layer_call_and_return_conditional_losses_17862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :
��2phi_2/kernel
:�2
phi_2/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_phi_3_layer_call_fn_17871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_phi_3_layer_call_and_return_conditional_losses_17882�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :
��2phi_3/kernel
:�2
phi_3/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_y0_hidden_1_layer_call_fn_17891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_17908�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#	�d2y0_hidden_1/kernel
:d2y0_hidden_1/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
'
d0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_y1_hidden_1_layer_call_fn_17917�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_17934�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#	�d2y1_hidden_1/kernel
:d2y1_hidden_1/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
'
e0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_y0_hidden_2_layer_call_fn_17943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_17960�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"dd2y0_hidden_2/kernel
:d2y0_hidden_2/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
'
f0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_y1_hidden_2_layer_call_fn_17969�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_17986�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"dd2y1_hidden_2/kernel
:d2y1_hidden_2/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_y0_predictions_layer_call_fn_17995�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_y0_predictions_layer_call_and_return_conditional_losses_18011�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%d2y0_predictions/kernel
!:2y0_predictions/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_y1_predictions_layer_call_fn_18020�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_y1_predictions_layer_call_and_return_conditional_losses_18036�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%d2y1_predictions/kernel
!:2y1_predictions/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_concatenate_layer_call_fn_18042�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_concatenate_layer_call_and_return_conditional_losses_18049�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
__inference_loss_fn_0_18060�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_18071�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_18082�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_18093�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_18104�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_18115�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_16937input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_17575inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_17616inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_17243input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_17719inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_17822inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_17329input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_17415input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
�B�
#__inference_signature_wrapper_17498input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_phi_1_layer_call_fn_17831inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_phi_1_layer_call_and_return_conditional_losses_17842inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_phi_2_layer_call_fn_17851inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_phi_2_layer_call_and_return_conditional_losses_17862inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_phi_3_layer_call_fn_17871inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_phi_3_layer_call_and_return_conditional_losses_17882inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_y0_hidden_1_layer_call_fn_17891inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_17908inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
d0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_y1_hidden_1_layer_call_fn_17917inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_17934inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
e0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_y0_hidden_2_layer_call_fn_17943inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_17960inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_y1_hidden_2_layer_call_fn_17969inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_17986inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_y0_predictions_layer_call_fn_17995inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_y0_predictions_layer_call_and_return_conditional_losses_18011inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_y1_predictions_layer_call_fn_18020inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_y1_predictions_layer_call_and_return_conditional_losses_18036inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_concatenate_layer_call_fn_18042inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_concatenate_layer_call_and_return_conditional_losses_18049inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_18060"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_18071"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_18082"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_18093"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_18104"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_18115"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
*:(	�2SGD/phi_1/kernel/momentum
$:"�2SGD/phi_1/bias/momentum
+:)
��2SGD/phi_2/kernel/momentum
$:"�2SGD/phi_2/bias/momentum
+:)
��2SGD/phi_3/kernel/momentum
$:"�2SGD/phi_3/bias/momentum
0:.	�d2SGD/y0_hidden_1/kernel/momentum
):'d2SGD/y0_hidden_1/bias/momentum
0:.	�d2SGD/y1_hidden_1/kernel/momentum
):'d2SGD/y1_hidden_1/bias/momentum
/:-dd2SGD/y0_hidden_2/kernel/momentum
):'d2SGD/y0_hidden_2/bias/momentum
/:-dd2SGD/y1_hidden_2/kernel/momentum
):'d2SGD/y1_hidden_2/bias/momentum
2:0d2"SGD/y0_predictions/kernel/momentum
,:*2 SGD/y0_predictions/bias/momentum
2:0d2"SGD/y1_predictions/kernel/momentum
,:*2 SGD/y1_predictions/bias/momentum�
 __inference__wrapped_model_16658#$+,;<34KLCDST[\.�+
$�!
�
input���������
� "9�6
4
concatenate%�"
concatenate����������
F__inference_concatenate_layer_call_and_return_conditional_losses_18049�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
+__inference_concatenate_layer_call_fn_18042vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "����������:
__inference_loss_fn_0_180603�

� 
� "� :
__inference_loss_fn_1_18071;�

� 
� "� :
__inference_loss_fn_2_18082C�

� 
� "� :
__inference_loss_fn_3_18093K�

� 
� "� :
__inference_loss_fn_4_18104S�

� 
� "� :
__inference_loss_fn_5_18115[�

� 
� "� �
@__inference_model_layer_call_and_return_conditional_losses_17329s#$+,;<34KLCDST[\6�3
,�)
�
input���������
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_17415s#$+,;<34KLCDST[\6�3
,�)
�
input���������
p

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_17719t#$+,;<34KLCDST[\7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_17822t#$+,;<34KLCDST[\7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
%__inference_model_layer_call_fn_16937f#$+,;<34KLCDST[\6�3
,�)
�
input���������
p 

 
� "�����������
%__inference_model_layer_call_fn_17243f#$+,;<34KLCDST[\6�3
,�)
�
input���������
p

 
� "�����������
%__inference_model_layer_call_fn_17575g#$+,;<34KLCDST[\7�4
-�*
 �
inputs���������
p 

 
� "�����������
%__inference_model_layer_call_fn_17616g#$+,;<34KLCDST[\7�4
-�*
 �
inputs���������
p

 
� "�����������
@__inference_phi_1_layer_call_and_return_conditional_losses_17842]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� y
%__inference_phi_1_layer_call_fn_17831P/�,
%�"
 �
inputs���������
� "������������
@__inference_phi_2_layer_call_and_return_conditional_losses_17862^#$0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
%__inference_phi_2_layer_call_fn_17851Q#$0�-
&�#
!�
inputs����������
� "������������
@__inference_phi_3_layer_call_and_return_conditional_losses_17882^+,0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
%__inference_phi_3_layer_call_fn_17871Q+,0�-
&�#
!�
inputs����������
� "������������
#__inference_signature_wrapper_17498�#$+,;<34KLCDST[\7�4
� 
-�*
(
input�
input���������"9�6
4
concatenate%�"
concatenate����������
F__inference_y0_hidden_1_layer_call_and_return_conditional_losses_17908]340�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� 
+__inference_y0_hidden_1_layer_call_fn_17891P340�-
&�#
!�
inputs����������
� "����������d�
F__inference_y0_hidden_2_layer_call_and_return_conditional_losses_17960\CD/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
+__inference_y0_hidden_2_layer_call_fn_17943OCD/�,
%�"
 �
inputs���������d
� "����������d�
I__inference_y0_predictions_layer_call_and_return_conditional_losses_18011\ST/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� �
.__inference_y0_predictions_layer_call_fn_17995OST/�,
%�"
 �
inputs���������d
� "�����������
F__inference_y1_hidden_1_layer_call_and_return_conditional_losses_17934];<0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� 
+__inference_y1_hidden_1_layer_call_fn_17917P;<0�-
&�#
!�
inputs����������
� "����������d�
F__inference_y1_hidden_2_layer_call_and_return_conditional_losses_17986\KL/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
+__inference_y1_hidden_2_layer_call_fn_17969OKL/�,
%�"
 �
inputs���������d
� "����������d�
I__inference_y1_predictions_layer_call_and_return_conditional_losses_18036\[\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� �
.__inference_y1_predictions_layer_call_fn_18020O[\/�,
%�"
 �
inputs���������d
� "����������