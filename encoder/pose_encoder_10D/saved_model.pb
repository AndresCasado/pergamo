??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*	2.2.0-rc12v2.2.0-rc0-43-gacf4951a2f8??	
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
 regularization_losses
!trainable_variables
"	keras_api

#0
$1
%2
&3
 

#0
$1
%2
&3
?
'layer_metrics
(layer_regularization_losses
)non_trainable_variables
*metrics
	variables
regularization_losses

+layers
	trainable_variables
 
 
 
 
?
,layer_metrics
-layer_regularization_losses
.non_trainable_variables
/metrics
	variables
regularization_losses

0layers
trainable_variables
 
 
 
?
1layer_metrics
2layer_regularization_losses
3non_trainable_variables
4metrics
	variables
regularization_losses

5layers
trainable_variables
 
 
 
?
6layer_metrics
7layer_regularization_losses
8non_trainable_variables
9metrics
	variables
regularization_losses

:layers
trainable_variables
 
 
 
?
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
>metrics
	variables
regularization_losses

?layers
trainable_variables
 
h

#kernel
$bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

%kernel
&bias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api

#0
$1
%2
&3
 

#0
$1
%2
&3
?
Hlayer_metrics
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
	variables
 regularization_losses

Llayers
!trainable_variables
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

#0
$1
 

#0
$1
?
Mlayer_metrics
Nlayer_regularization_losses
Onon_trainable_variables
Pmetrics
@	variables
Aregularization_losses

Qlayers
Btrainable_variables

%0
&1
 

%0
&1
?
Rlayer_metrics
Slayer_regularization_losses
Tnon_trainable_variables
Umetrics
D	variables
Eregularization_losses

Vlayers
Ftrainable_variables
 
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
x
serving_default_posesPlaceholder*'
_output_shapes
:?????????E*
dtype0*
shape:?????????E
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_posesdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*,
f'R%
#__inference_signature_wrapper_56584
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*'
f"R 
__inference__traced_save_57142
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8**
f%R#
!__inference__traced_restore_57166??	
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_56709

inputs5
1pose_encoder_dense_matmul_readvariableop_resource6
2pose_encoder_dense_biasadd_readvariableop_resource7
3pose_encoder_dense_1_matmul_readvariableop_resource8
4pose_encoder_dense_1_biasadd_readvariableop_resource
identity??
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2#
!tf_op_layer_Reshape/Reshape/shape?
tf_op_layer_Reshape/ReshapeReshapeinputs*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*+
_output_shapes
:?????????2
tf_op_layer_Reshape/Reshape?
axis_angle_to_matrix/ShapeShape$tf_op_layer_Reshape/Reshape:output:0*
T0*
_output_shapes
:2
axis_angle_to_matrix/Shape?
"axis_angle_to_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2$
"axis_angle_to_matrix/Reshape/shape?
axis_angle_to_matrix/ReshapeReshape$tf_op_layer_Reshape/Reshape:output:0+axis_angle_to_matrix/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/Reshape?
axis_angle_to_matrix/Shape_1Shape%axis_angle_to_matrix/Reshape:output:0*
T0*
_output_shapes
:2
axis_angle_to_matrix/Shape_1?
(axis_angle_to_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(axis_angle_to_matrix/strided_slice/stack?
*axis_angle_to_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*axis_angle_to_matrix/strided_slice/stack_1?
*axis_angle_to_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*axis_angle_to_matrix/strided_slice/stack_2?
"axis_angle_to_matrix/strided_sliceStridedSlice%axis_angle_to_matrix/Shape_1:output:01axis_angle_to_matrix/strided_slice/stack:output:03axis_angle_to_matrix/strided_slice/stack_1:output:03axis_angle_to_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"axis_angle_to_matrix/strided_slice}
axis_angle_to_matrix/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
axis_angle_to_matrix/add/y?
axis_angle_to_matrix/addAddV2%axis_angle_to_matrix/Reshape:output:0#axis_angle_to_matrix/add/y:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/add?
axis_angle_to_matrix/norm/mulMulaxis_angle_to_matrix/add:z:0axis_angle_to_matrix/add:z:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/norm/mul?
/axis_angle_to_matrix/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:21
/axis_angle_to_matrix/norm/Sum/reduction_indices?
axis_angle_to_matrix/norm/SumSum!axis_angle_to_matrix/norm/mul:z:08axis_angle_to_matrix/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
axis_angle_to_matrix/norm/Sum?
axis_angle_to_matrix/norm/SqrtSqrt&axis_angle_to_matrix/norm/Sum:output:0*
T0*'
_output_shapes
:?????????2 
axis_angle_to_matrix/norm/Sqrt?
!axis_angle_to_matrix/norm/SqueezeSqueeze"axis_angle_to_matrix/norm/Sqrt:y:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
2#
!axis_angle_to_matrix/norm/Squeeze?
#axis_angle_to_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#axis_angle_to_matrix/ExpandDims/dim?
axis_angle_to_matrix/ExpandDims
ExpandDims*axis_angle_to_matrix/norm/Squeeze:output:0,axis_angle_to_matrix/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2!
axis_angle_to_matrix/ExpandDims?
axis_angle_to_matrix/truedivRealDiv%axis_angle_to_matrix/Reshape:output:0(axis_angle_to_matrix/ExpandDims:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/truediv?
%axis_angle_to_matrix/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%axis_angle_to_matrix/ExpandDims_1/dim?
!axis_angle_to_matrix/ExpandDims_1
ExpandDims axis_angle_to_matrix/truediv:z:0.axis_angle_to_matrix/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2#
!axis_angle_to_matrix/ExpandDims_1?
%axis_angle_to_matrix/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%axis_angle_to_matrix/ExpandDims_2/dim?
!axis_angle_to_matrix/ExpandDims_2
ExpandDims(axis_angle_to_matrix/ExpandDims:output:0.axis_angle_to_matrix/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:?????????2#
!axis_angle_to_matrix/ExpandDims_2?
axis_angle_to_matrix/CosCos*axis_angle_to_matrix/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/Cos?
axis_angle_to_matrix/SinSin*axis_angle_to_matrix/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/Sin?
axis_angle_to_matrix/outerBatchMatMulV2*axis_angle_to_matrix/ExpandDims_1:output:0*axis_angle_to_matrix/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????*
adj_y(2
axis_angle_to_matrix/outer?
axis_angle_to_matrix/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ??2
axis_angle_to_matrix/eye/ones?
axis_angle_to_matrix/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2!
axis_angle_to_matrix/eye/diag/k?
&axis_angle_to_matrix/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&axis_angle_to_matrix/eye/diag/num_rows?
&axis_angle_to_matrix/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&axis_angle_to_matrix/eye/diag/num_cols?
+axis_angle_to_matrix/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+axis_angle_to_matrix/eye/diag/padding_value?
axis_angle_to_matrix/eye/diagMatrixDiagV3&axis_angle_to_matrix/eye/ones:output:0(axis_angle_to_matrix/eye/diag/k:output:0/axis_angle_to_matrix/eye/diag/num_rows:output:0/axis_angle_to_matrix/eye/diag/num_cols:output:04axis_angle_to_matrix/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2
axis_angle_to_matrix/eye/diag?
%axis_angle_to_matrix/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%axis_angle_to_matrix/ExpandDims_3/dim?
!axis_angle_to_matrix/ExpandDims_3
ExpandDims&axis_angle_to_matrix/eye/diag:output:0.axis_angle_to_matrix/ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2#
!axis_angle_to_matrix/ExpandDims_3?
%axis_angle_to_matrix/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%axis_angle_to_matrix/Tile/multiples/1?
%axis_angle_to_matrix/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%axis_angle_to_matrix/Tile/multiples/2?
#axis_angle_to_matrix/Tile/multiplesPack+axis_angle_to_matrix/strided_slice:output:0.axis_angle_to_matrix/Tile/multiples/1:output:0.axis_angle_to_matrix/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:2%
#axis_angle_to_matrix/Tile/multiples?
axis_angle_to_matrix/TileTile*axis_angle_to_matrix/ExpandDims_3:output:0,axis_angle_to_matrix/Tile/multiples:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/Tile?
axis_angle_to_matrix/mulMulaxis_angle_to_matrix/Cos:y:0"axis_angle_to_matrix/Tile:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/mul}
axis_angle_to_matrix/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
axis_angle_to_matrix/sub/x?
axis_angle_to_matrix/subSub#axis_angle_to_matrix/sub/x:output:0axis_angle_to_matrix/Cos:y:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/sub?
axis_angle_to_matrix/mul_1Mulaxis_angle_to_matrix/sub:z:0#axis_angle_to_matrix/outer:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/mul_1?
axis_angle_to_matrix/add_1AddV2axis_angle_to_matrix/mul:z:0axis_angle_to_matrix/mul_1:z:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/add_1?
axis_angle_to_matrix/skew/ShapeShape*axis_angle_to_matrix/ExpandDims_1:output:0*
T0*
_output_shapes
:2!
axis_angle_to_matrix/skew/Shape?
-axis_angle_to_matrix/skew/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-axis_angle_to_matrix/skew/strided_slice/stack?
/axis_angle_to_matrix/skew/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/axis_angle_to_matrix/skew/strided_slice/stack_1?
/axis_angle_to_matrix/skew/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/axis_angle_to_matrix/skew/strided_slice/stack_2?
'axis_angle_to_matrix/skew/strided_sliceStridedSlice(axis_angle_to_matrix/skew/Shape:output:06axis_angle_to_matrix/skew/strided_slice/stack:output:08axis_angle_to_matrix/skew/strided_slice/stack_1:output:08axis_angle_to_matrix/skew/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'axis_angle_to_matrix/skew/strided_slice?
axis_angle_to_matrix/skew/ConstConst*
_output_shapes
:*
dtype0*-
value$B""                  2!
axis_angle_to_matrix/skew/Const?
%axis_angle_to_matrix/skew/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2'
%axis_angle_to_matrix/skew/range/start?
%axis_angle_to_matrix/skew/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2'
%axis_angle_to_matrix/skew/range/delta?
axis_angle_to_matrix/skew/rangeRange.axis_angle_to_matrix/skew/range/start:output:00axis_angle_to_matrix/skew/strided_slice:output:0.axis_angle_to_matrix/skew/range/delta:output:0*#
_output_shapes
:?????????2!
axis_angle_to_matrix/skew/range?
axis_angle_to_matrix/skew/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2!
axis_angle_to_matrix/skew/mul/y?
axis_angle_to_matrix/skew/mulMul(axis_angle_to_matrix/skew/range:output:0(axis_angle_to_matrix/skew/mul/y:output:0*
T0*#
_output_shapes
:?????????2
axis_angle_to_matrix/skew/mul?
'axis_angle_to_matrix/skew/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2)
'axis_angle_to_matrix/skew/Reshape/shape?
!axis_angle_to_matrix/skew/ReshapeReshape!axis_angle_to_matrix/skew/mul:z:00axis_angle_to_matrix/skew/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2#
!axis_angle_to_matrix/skew/Reshape?
axis_angle_to_matrix/skew/addAddV2*axis_angle_to_matrix/skew/Reshape:output:0(axis_angle_to_matrix/skew/Const:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/skew/add?
)axis_angle_to_matrix/skew/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)axis_angle_to_matrix/skew/Reshape_1/shape?
#axis_angle_to_matrix/skew/Reshape_1Reshape!axis_angle_to_matrix/skew/add:z:02axis_angle_to_matrix/skew/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/Reshape_1?
/axis_angle_to_matrix/skew/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_1/stack?
1axis_angle_to_matrix/skew/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_1/stack_1?
1axis_angle_to_matrix/skew/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_1/stack_2?
)axis_angle_to_matrix/skew/strided_slice_1StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_1/stack:output:0:axis_angle_to_matrix/skew/strided_slice_1/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_1?
axis_angle_to_matrix/skew/NegNeg2axis_angle_to_matrix/skew/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/skew/Neg?
/axis_angle_to_matrix/skew/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_2/stack?
1axis_angle_to_matrix/skew/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_2/stack_1?
1axis_angle_to_matrix/skew/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_2/stack_2?
)axis_angle_to_matrix/skew/strided_slice_2StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_2/stack:output:0:axis_angle_to_matrix/skew/strided_slice_2/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_2?
/axis_angle_to_matrix/skew/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_3/stack?
1axis_angle_to_matrix/skew/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_3/stack_1?
1axis_angle_to_matrix/skew/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_3/stack_2?
)axis_angle_to_matrix/skew/strided_slice_3StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_3/stack:output:0:axis_angle_to_matrix/skew/strided_slice_3/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_3?
/axis_angle_to_matrix/skew/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/axis_angle_to_matrix/skew/strided_slice_4/stack?
1axis_angle_to_matrix/skew/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_4/stack_1?
1axis_angle_to_matrix/skew/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_4/stack_2?
)axis_angle_to_matrix/skew/strided_slice_4StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_4/stack:output:0:axis_angle_to_matrix/skew/strided_slice_4/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_4?
axis_angle_to_matrix/skew/Neg_1Neg2axis_angle_to_matrix/skew/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2!
axis_angle_to_matrix/skew/Neg_1?
/axis_angle_to_matrix/skew/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_5/stack?
1axis_angle_to_matrix/skew/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_5/stack_1?
1axis_angle_to_matrix/skew/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_5/stack_2?
)axis_angle_to_matrix/skew/strided_slice_5StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_5/stack:output:0:axis_angle_to_matrix/skew/strided_slice_5/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_5?
axis_angle_to_matrix/skew/Neg_2Neg2axis_angle_to_matrix/skew/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2!
axis_angle_to_matrix/skew/Neg_2?
/axis_angle_to_matrix/skew/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/axis_angle_to_matrix/skew/strided_slice_6/stack?
1axis_angle_to_matrix/skew/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_6/stack_1?
1axis_angle_to_matrix/skew/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_6/stack_2?
)axis_angle_to_matrix/skew/strided_slice_6StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_6/stack:output:0:axis_angle_to_matrix/skew/strided_slice_6/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_6?
axis_angle_to_matrix/skew/stackPack!axis_angle_to_matrix/skew/Neg:y:02axis_angle_to_matrix/skew/strided_slice_2:output:02axis_angle_to_matrix/skew/strided_slice_3:output:0#axis_angle_to_matrix/skew/Neg_1:y:0#axis_angle_to_matrix/skew/Neg_2:y:02axis_angle_to_matrix/skew/strided_slice_6:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2!
axis_angle_to_matrix/skew/stack?
)axis_angle_to_matrix/skew/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)axis_angle_to_matrix/skew/Reshape_2/shape?
#axis_angle_to_matrix/skew/Reshape_2Reshape(axis_angle_to_matrix/skew/stack:output:02axis_angle_to_matrix/skew/Reshape_2/shape:output:0*
T0*#
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/Reshape_2?
!axis_angle_to_matrix/skew/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :	2#
!axis_angle_to_matrix/skew/mul_1/y?
axis_angle_to_matrix/skew/mul_1Mul0axis_angle_to_matrix/skew/strided_slice:output:0*axis_angle_to_matrix/skew/mul_1/y:output:0*
T0*
_output_shapes
: 2!
axis_angle_to_matrix/skew/mul_1?
)axis_angle_to_matrix/skew/ScatterNd/shapePack#axis_angle_to_matrix/skew/mul_1:z:0*
N*
T0*
_output_shapes
:2+
)axis_angle_to_matrix/skew/ScatterNd/shape?
#axis_angle_to_matrix/skew/ScatterNd	ScatterNd,axis_angle_to_matrix/skew/Reshape_1:output:0,axis_angle_to_matrix/skew/Reshape_2:output:02axis_angle_to_matrix/skew/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/ScatterNd?
+axis_angle_to_matrix/skew/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+axis_angle_to_matrix/skew/Reshape_3/shape/1?
+axis_angle_to_matrix/skew/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+axis_angle_to_matrix/skew/Reshape_3/shape/2?
)axis_angle_to_matrix/skew/Reshape_3/shapePack0axis_angle_to_matrix/skew/strided_slice:output:04axis_angle_to_matrix/skew/Reshape_3/shape/1:output:04axis_angle_to_matrix/skew/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)axis_angle_to_matrix/skew/Reshape_3/shape?
#axis_angle_to_matrix/skew/Reshape_3Reshape,axis_angle_to_matrix/skew/ScatterNd:output:02axis_angle_to_matrix/skew/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/Reshape_3?
axis_angle_to_matrix/mul_2Mulaxis_angle_to_matrix/Sin:y:0,axis_angle_to_matrix/skew/Reshape_3:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/mul_2?
axis_angle_to_matrix/add_2AddV2axis_angle_to_matrix/add_1:z:0axis_angle_to_matrix/mul_2:z:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/add_2?
$axis_angle_to_matrix/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$axis_angle_to_matrix/concat/values_1?
 axis_angle_to_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 axis_angle_to_matrix/concat/axis?
axis_angle_to_matrix/concatConcatV2#axis_angle_to_matrix/Shape:output:0-axis_angle_to_matrix/concat/values_1:output:0)axis_angle_to_matrix/concat/axis:output:0*
N*
T0*
_output_shapes
:2
axis_angle_to_matrix/concat?
axis_angle_to_matrix/Reshape_1Reshapeaxis_angle_to_matrix/add_2:z:0$axis_angle_to_matrix/concat:output:0*
T0*/
_output_shapes
:?????????2 
axis_angle_to_matrix/Reshape_1?
tf_op_layer_Sub/Sub/yConst*
_output_shapes

:*
dtype0*=
value4B2"$  ??              ??              ??2
tf_op_layer_Sub/Sub/y?
tf_op_layer_Sub/SubSub'axis_angle_to_matrix/Reshape_1:output:0tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????2
tf_op_layer_Sub/Sub?
%tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%tf_op_layer_Reshape_1/Reshape_1/shape?
tf_op_layer_Reshape_1/Reshape_1Reshapetf_op_layer_Sub/Sub:z:0.tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*(
_output_shapes
:??????????2!
tf_op_layer_Reshape_1/Reshape_1?
(pose_encoder/dense/MatMul/ReadVariableOpReadVariableOp1pose_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(pose_encoder/dense/MatMul/ReadVariableOp?
pose_encoder/dense/MatMulMatMul(tf_op_layer_Reshape_1/Reshape_1:output:00pose_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
pose_encoder/dense/MatMul?
)pose_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2pose_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)pose_encoder/dense/BiasAdd/ReadVariableOp?
pose_encoder/dense/BiasAddBiasAdd#pose_encoder/dense/MatMul:product:01pose_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
pose_encoder/dense/BiasAdd?
pose_encoder/dense/ReluRelu#pose_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
pose_encoder/dense/Relu?
*pose_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3pose_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*pose_encoder/dense_1/MatMul/ReadVariableOp?
pose_encoder/dense_1/MatMulMatMul%pose_encoder/dense/Relu:activations:02pose_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
pose_encoder/dense_1/MatMul?
+pose_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4pose_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+pose_encoder/dense_1/BiasAdd/ReadVariableOp?
pose_encoder/dense_1/BiasAddBiasAdd%pose_encoder/dense_1/MatMul:product:03pose_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
pose_encoder/dense_1/BiasAddy
IdentityIdentity%pose_encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E:::::O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
Q
5__inference_tf_op_layer_Reshape_1_layer_call_fn_57004

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_564472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?q
o
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_56419

axis_angle
identityH
ShapeShape
axis_angle*
T0*
_output_shapes
:2
Shapeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapes
ReshapeReshape
axis_angleReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
ReshapeR
Shape_1ShapeReshape:output:0*
T0*
_output_shapes
:2	
Shape_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/yg
addAddV2Reshape:output:0add/y:output:0*
T0*'
_output_shapes
:?????????2
add_
norm/mulMuladd:z:0add:z:0*
T0*'
_output_shapes
:?????????2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt{
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
2
norm/Squeezek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsnorm/Squeeze:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsv
truedivRealDivReshape:output:0ExpandDims:output:0*
T0*'
_output_shapes
:?????????2	
truedivo
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimstruediv:z:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDimsExpandDims:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_2^
CosCosExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
Cos^
SinSinExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
Sin?
outerBatchMatMulV2ExpandDims_1:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????*
adj_y(2
outera
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ??2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagf
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_3/dim?
ExpandDims_3
ExpandDimseye/diag:output:0ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_3f
Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :2
Tile/multiples/1f
Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :2
Tile/multiples/2?
Tile/multiplesPackstrided_slice:output:0Tile/multiples/1:output:0Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:2
Tile/multiplesz
TileTileExpandDims_3:output:0Tile/multiples:output:0*
T0*+
_output_shapes
:?????????2
Tile_
mulMulCos:y:0Tile:output:0*
T0*+
_output_shapes
:?????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Cos:y:0*
T0*+
_output_shapes
:?????????2
subd
mul_1Mulsub:z:0outer:output:0*
T0*+
_output_shapes
:?????????2
mul_1a
add_1AddV2mul:z:0	mul_1:z:0*
T0*+
_output_shapes
:?????????2
add_1]

skew/ShapeShapeExpandDims_1:output:0*
T0*
_output_shapes
:2

skew/Shape~
skew/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
skew/strided_slice/stack?
skew/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
skew/strided_slice/stack_1?
skew/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
skew/strided_slice/stack_2?
skew/strided_sliceStridedSliceskew/Shape:output:0!skew/strided_slice/stack:output:0#skew/strided_slice/stack_1:output:0#skew/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
skew/strided_slicey

skew/ConstConst*
_output_shapes
:*
dtype0*-
value$B""                  2

skew/Constf
skew/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
skew/range/startf
skew/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
skew/range/delta?

skew/rangeRangeskew/range/start:output:0skew/strided_slice:output:0skew/range/delta:output:0*#
_output_shapes
:?????????2

skew/rangeZ

skew/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2

skew/mul/ys
skew/mulMulskew/range:output:0skew/mul/y:output:0*
T0*#
_output_shapes
:?????????2

skew/muly
skew/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
skew/Reshape/shape?
skew/ReshapeReshapeskew/mul:z:0skew/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
skew/Reshape{
skew/addAddV2skew/Reshape:output:0skew/Const:output:0*
T0*'
_output_shapes
:?????????2

skew/add}
skew/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
skew/Reshape_1/shape?
skew/Reshape_1Reshapeskew/add:z:0skew/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
skew/Reshape_1?
skew/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_1/stack?
skew/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_1/stack_1?
skew/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_1/stack_2?
skew/strided_slice_1StridedSliceExpandDims_1:output:0#skew/strided_slice_1/stack:output:0%skew/strided_slice_1/stack_1:output:0%skew/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_1l
skew/NegNegskew/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

skew/Neg?
skew/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_2/stack?
skew/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_2/stack_1?
skew/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_2/stack_2?
skew/strided_slice_2StridedSliceExpandDims_1:output:0#skew/strided_slice_2/stack:output:0%skew/strided_slice_2/stack_1:output:0%skew/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_2?
skew/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_3/stack?
skew/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_3/stack_1?
skew/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_3/stack_2?
skew/strided_slice_3StridedSliceExpandDims_1:output:0#skew/strided_slice_3/stack:output:0%skew/strided_slice_3/stack_1:output:0%skew/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_3?
skew/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
skew/strided_slice_4/stack?
skew/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_4/stack_1?
skew/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_4/stack_2?
skew/strided_slice_4StridedSliceExpandDims_1:output:0#skew/strided_slice_4/stack:output:0%skew/strided_slice_4/stack_1:output:0%skew/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_4p

skew/Neg_1Negskew/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

skew/Neg_1?
skew/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_5/stack?
skew/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_5/stack_1?
skew/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_5/stack_2?
skew/strided_slice_5StridedSliceExpandDims_1:output:0#skew/strided_slice_5/stack:output:0%skew/strided_slice_5/stack_1:output:0%skew/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_5p

skew/Neg_2Negskew/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

skew/Neg_2?
skew/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
skew/strided_slice_6/stack?
skew/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_6/stack_1?
skew/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_6/stack_2?
skew/strided_slice_6StridedSliceExpandDims_1:output:0#skew/strided_slice_6/stack:output:0%skew/strided_slice_6/stack_1:output:0%skew/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_6?

skew/stackPackskew/Neg:y:0skew/strided_slice_2:output:0skew/strided_slice_3:output:0skew/Neg_1:y:0skew/Neg_2:y:0skew/strided_slice_6:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2

skew/stack
skew/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
skew/Reshape_2/shape?
skew/Reshape_2Reshapeskew/stack:output:0skew/Reshape_2/shape:output:0*
T0*#
_output_shapes
:?????????2
skew/Reshape_2^
skew/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :	2
skew/mul_1/yt

skew/mul_1Mulskew/strided_slice:output:0skew/mul_1/y:output:0*
T0*
_output_shapes
: 2

skew/mul_1r
skew/ScatterNd/shapePackskew/mul_1:z:0*
N*
T0*
_output_shapes
:2
skew/ScatterNd/shape?
skew/ScatterNd	ScatterNdskew/Reshape_1:output:0skew/Reshape_2:output:0skew/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
skew/ScatterNdr
skew/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
skew/Reshape_3/shape/1r
skew/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
skew/Reshape_3/shape/2?
skew/Reshape_3/shapePackskew/strided_slice:output:0skew/Reshape_3/shape/1:output:0skew/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
skew/Reshape_3/shape?
skew/Reshape_3Reshapeskew/ScatterNd:output:0skew/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2
skew/Reshape_3m
mul_2MulSin:y:0skew/Reshape_3:output:0*
T0*+
_output_shapes
:?????????2
mul_2c
add_2AddV2	add_1:z:0	mul_2:z:0*
T0*+
_output_shapes
:?????????2
add_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Shape:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatw
	Reshape_1Reshape	add_2:z:0concat:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_1n
IdentityIdentityReshape_1:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:W S
+
_output_shapes
:?????????
$
_user_specified_name
axis_angle
?
z
%__inference_dense_layer_call_fn_57084

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_561832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_56508	
poses
pose_encoder_56498
pose_encoder_56500
pose_encoder_56502
pose_encoder_56504
identity??$pose_encoder/StatefulPartitionedCall?
#tf_op_layer_Reshape/PartitionedCallPartitionedCallposes*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_563052%
#tf_op_layer_Reshape/PartitionedCall?
$axis_angle_to_matrix/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*X
fSRQ
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_564192&
$axis_angle_to_matrix/PartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall-axis_angle_to_matrix/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_564332!
tf_op_layer_Sub/PartitionedCall?
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_564472'
%tf_op_layer_Reshape_1/PartitionedCall?
$pose_encoder/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0pose_encoder_56498pose_encoder_56500pose_encoder_56502pose_encoder_56504*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562842&
$pose_encoder/StatefulPartitionedCall?
IdentityIdentity-pose_encoder/StatefulPartitionedCall:output:0%^pose_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::2L
$pose_encoder/StatefulPartitionedCall$pose_encoder/StatefulPartitionedCall:N J
'
_output_shapes
:?????????E

_user_specified_nameposes:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_57075

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
K
/__inference_tf_op_layer_Sub_layer_call_fn_56993

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_564332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_pose_encoder_layer_call_fn_57051

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_56305

inputs
identitys
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????E:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs
?
T
4__inference_axis_angle_to_matrix_layer_call_fn_56982

axis_angle
identity?
PartitionedCallPartitionedCall
axis_angle*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*X
fSRQ
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_564192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:W S
+
_output_shapes
:?????????
$
_user_specified_name
axis_angle
?
f
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_56988

inputs
identity?
Sub/yConst*
_output_shapes

:*
dtype0*=
value4B2"$  ??              ??              ??2
Sub/yr
SubSubinputsSub/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????2
Subc
IdentityIdentitySub:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56226
input_3
dense_56194
dense_56196
dense_1_56220
dense_1_56222
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_56194dense_56196*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_561832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56220dense_1_56222*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_562092!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_56834

inputs5
1pose_encoder_dense_matmul_readvariableop_resource6
2pose_encoder_dense_biasadd_readvariableop_resource7
3pose_encoder_dense_1_matmul_readvariableop_resource8
4pose_encoder_dense_1_biasadd_readvariableop_resource
identity??
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2#
!tf_op_layer_Reshape/Reshape/shape?
tf_op_layer_Reshape/ReshapeReshapeinputs*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*+
_output_shapes
:?????????2
tf_op_layer_Reshape/Reshape?
axis_angle_to_matrix/ShapeShape$tf_op_layer_Reshape/Reshape:output:0*
T0*
_output_shapes
:2
axis_angle_to_matrix/Shape?
"axis_angle_to_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2$
"axis_angle_to_matrix/Reshape/shape?
axis_angle_to_matrix/ReshapeReshape$tf_op_layer_Reshape/Reshape:output:0+axis_angle_to_matrix/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/Reshape?
axis_angle_to_matrix/Shape_1Shape%axis_angle_to_matrix/Reshape:output:0*
T0*
_output_shapes
:2
axis_angle_to_matrix/Shape_1?
(axis_angle_to_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(axis_angle_to_matrix/strided_slice/stack?
*axis_angle_to_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*axis_angle_to_matrix/strided_slice/stack_1?
*axis_angle_to_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*axis_angle_to_matrix/strided_slice/stack_2?
"axis_angle_to_matrix/strided_sliceStridedSlice%axis_angle_to_matrix/Shape_1:output:01axis_angle_to_matrix/strided_slice/stack:output:03axis_angle_to_matrix/strided_slice/stack_1:output:03axis_angle_to_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"axis_angle_to_matrix/strided_slice}
axis_angle_to_matrix/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
axis_angle_to_matrix/add/y?
axis_angle_to_matrix/addAddV2%axis_angle_to_matrix/Reshape:output:0#axis_angle_to_matrix/add/y:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/add?
axis_angle_to_matrix/norm/mulMulaxis_angle_to_matrix/add:z:0axis_angle_to_matrix/add:z:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/norm/mul?
/axis_angle_to_matrix/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:21
/axis_angle_to_matrix/norm/Sum/reduction_indices?
axis_angle_to_matrix/norm/SumSum!axis_angle_to_matrix/norm/mul:z:08axis_angle_to_matrix/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
axis_angle_to_matrix/norm/Sum?
axis_angle_to_matrix/norm/SqrtSqrt&axis_angle_to_matrix/norm/Sum:output:0*
T0*'
_output_shapes
:?????????2 
axis_angle_to_matrix/norm/Sqrt?
!axis_angle_to_matrix/norm/SqueezeSqueeze"axis_angle_to_matrix/norm/Sqrt:y:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
2#
!axis_angle_to_matrix/norm/Squeeze?
#axis_angle_to_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#axis_angle_to_matrix/ExpandDims/dim?
axis_angle_to_matrix/ExpandDims
ExpandDims*axis_angle_to_matrix/norm/Squeeze:output:0,axis_angle_to_matrix/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2!
axis_angle_to_matrix/ExpandDims?
axis_angle_to_matrix/truedivRealDiv%axis_angle_to_matrix/Reshape:output:0(axis_angle_to_matrix/ExpandDims:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/truediv?
%axis_angle_to_matrix/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%axis_angle_to_matrix/ExpandDims_1/dim?
!axis_angle_to_matrix/ExpandDims_1
ExpandDims axis_angle_to_matrix/truediv:z:0.axis_angle_to_matrix/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2#
!axis_angle_to_matrix/ExpandDims_1?
%axis_angle_to_matrix/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%axis_angle_to_matrix/ExpandDims_2/dim?
!axis_angle_to_matrix/ExpandDims_2
ExpandDims(axis_angle_to_matrix/ExpandDims:output:0.axis_angle_to_matrix/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:?????????2#
!axis_angle_to_matrix/ExpandDims_2?
axis_angle_to_matrix/CosCos*axis_angle_to_matrix/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/Cos?
axis_angle_to_matrix/SinSin*axis_angle_to_matrix/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/Sin?
axis_angle_to_matrix/outerBatchMatMulV2*axis_angle_to_matrix/ExpandDims_1:output:0*axis_angle_to_matrix/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????*
adj_y(2
axis_angle_to_matrix/outer?
axis_angle_to_matrix/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ??2
axis_angle_to_matrix/eye/ones?
axis_angle_to_matrix/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2!
axis_angle_to_matrix/eye/diag/k?
&axis_angle_to_matrix/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&axis_angle_to_matrix/eye/diag/num_rows?
&axis_angle_to_matrix/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&axis_angle_to_matrix/eye/diag/num_cols?
+axis_angle_to_matrix/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+axis_angle_to_matrix/eye/diag/padding_value?
axis_angle_to_matrix/eye/diagMatrixDiagV3&axis_angle_to_matrix/eye/ones:output:0(axis_angle_to_matrix/eye/diag/k:output:0/axis_angle_to_matrix/eye/diag/num_rows:output:0/axis_angle_to_matrix/eye/diag/num_cols:output:04axis_angle_to_matrix/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2
axis_angle_to_matrix/eye/diag?
%axis_angle_to_matrix/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%axis_angle_to_matrix/ExpandDims_3/dim?
!axis_angle_to_matrix/ExpandDims_3
ExpandDims&axis_angle_to_matrix/eye/diag:output:0.axis_angle_to_matrix/ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2#
!axis_angle_to_matrix/ExpandDims_3?
%axis_angle_to_matrix/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%axis_angle_to_matrix/Tile/multiples/1?
%axis_angle_to_matrix/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%axis_angle_to_matrix/Tile/multiples/2?
#axis_angle_to_matrix/Tile/multiplesPack+axis_angle_to_matrix/strided_slice:output:0.axis_angle_to_matrix/Tile/multiples/1:output:0.axis_angle_to_matrix/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:2%
#axis_angle_to_matrix/Tile/multiples?
axis_angle_to_matrix/TileTile*axis_angle_to_matrix/ExpandDims_3:output:0,axis_angle_to_matrix/Tile/multiples:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/Tile?
axis_angle_to_matrix/mulMulaxis_angle_to_matrix/Cos:y:0"axis_angle_to_matrix/Tile:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/mul}
axis_angle_to_matrix/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
axis_angle_to_matrix/sub/x?
axis_angle_to_matrix/subSub#axis_angle_to_matrix/sub/x:output:0axis_angle_to_matrix/Cos:y:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/sub?
axis_angle_to_matrix/mul_1Mulaxis_angle_to_matrix/sub:z:0#axis_angle_to_matrix/outer:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/mul_1?
axis_angle_to_matrix/add_1AddV2axis_angle_to_matrix/mul:z:0axis_angle_to_matrix/mul_1:z:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/add_1?
axis_angle_to_matrix/skew/ShapeShape*axis_angle_to_matrix/ExpandDims_1:output:0*
T0*
_output_shapes
:2!
axis_angle_to_matrix/skew/Shape?
-axis_angle_to_matrix/skew/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-axis_angle_to_matrix/skew/strided_slice/stack?
/axis_angle_to_matrix/skew/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/axis_angle_to_matrix/skew/strided_slice/stack_1?
/axis_angle_to_matrix/skew/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/axis_angle_to_matrix/skew/strided_slice/stack_2?
'axis_angle_to_matrix/skew/strided_sliceStridedSlice(axis_angle_to_matrix/skew/Shape:output:06axis_angle_to_matrix/skew/strided_slice/stack:output:08axis_angle_to_matrix/skew/strided_slice/stack_1:output:08axis_angle_to_matrix/skew/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'axis_angle_to_matrix/skew/strided_slice?
axis_angle_to_matrix/skew/ConstConst*
_output_shapes
:*
dtype0*-
value$B""                  2!
axis_angle_to_matrix/skew/Const?
%axis_angle_to_matrix/skew/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2'
%axis_angle_to_matrix/skew/range/start?
%axis_angle_to_matrix/skew/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2'
%axis_angle_to_matrix/skew/range/delta?
axis_angle_to_matrix/skew/rangeRange.axis_angle_to_matrix/skew/range/start:output:00axis_angle_to_matrix/skew/strided_slice:output:0.axis_angle_to_matrix/skew/range/delta:output:0*#
_output_shapes
:?????????2!
axis_angle_to_matrix/skew/range?
axis_angle_to_matrix/skew/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2!
axis_angle_to_matrix/skew/mul/y?
axis_angle_to_matrix/skew/mulMul(axis_angle_to_matrix/skew/range:output:0(axis_angle_to_matrix/skew/mul/y:output:0*
T0*#
_output_shapes
:?????????2
axis_angle_to_matrix/skew/mul?
'axis_angle_to_matrix/skew/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2)
'axis_angle_to_matrix/skew/Reshape/shape?
!axis_angle_to_matrix/skew/ReshapeReshape!axis_angle_to_matrix/skew/mul:z:00axis_angle_to_matrix/skew/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2#
!axis_angle_to_matrix/skew/Reshape?
axis_angle_to_matrix/skew/addAddV2*axis_angle_to_matrix/skew/Reshape:output:0(axis_angle_to_matrix/skew/Const:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/skew/add?
)axis_angle_to_matrix/skew/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)axis_angle_to_matrix/skew/Reshape_1/shape?
#axis_angle_to_matrix/skew/Reshape_1Reshape!axis_angle_to_matrix/skew/add:z:02axis_angle_to_matrix/skew/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/Reshape_1?
/axis_angle_to_matrix/skew/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_1/stack?
1axis_angle_to_matrix/skew/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_1/stack_1?
1axis_angle_to_matrix/skew/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_1/stack_2?
)axis_angle_to_matrix/skew/strided_slice_1StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_1/stack:output:0:axis_angle_to_matrix/skew/strided_slice_1/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_1?
axis_angle_to_matrix/skew/NegNeg2axis_angle_to_matrix/skew/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
axis_angle_to_matrix/skew/Neg?
/axis_angle_to_matrix/skew/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_2/stack?
1axis_angle_to_matrix/skew/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_2/stack_1?
1axis_angle_to_matrix/skew/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_2/stack_2?
)axis_angle_to_matrix/skew/strided_slice_2StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_2/stack:output:0:axis_angle_to_matrix/skew/strided_slice_2/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_2?
/axis_angle_to_matrix/skew/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_3/stack?
1axis_angle_to_matrix/skew/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_3/stack_1?
1axis_angle_to_matrix/skew/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_3/stack_2?
)axis_angle_to_matrix/skew/strided_slice_3StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_3/stack:output:0:axis_angle_to_matrix/skew/strided_slice_3/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_3?
/axis_angle_to_matrix/skew/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/axis_angle_to_matrix/skew/strided_slice_4/stack?
1axis_angle_to_matrix/skew/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_4/stack_1?
1axis_angle_to_matrix/skew/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_4/stack_2?
)axis_angle_to_matrix/skew/strided_slice_4StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_4/stack:output:0:axis_angle_to_matrix/skew/strided_slice_4/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_4?
axis_angle_to_matrix/skew/Neg_1Neg2axis_angle_to_matrix/skew/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2!
axis_angle_to_matrix/skew/Neg_1?
/axis_angle_to_matrix/skew/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/axis_angle_to_matrix/skew/strided_slice_5/stack?
1axis_angle_to_matrix/skew/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_5/stack_1?
1axis_angle_to_matrix/skew/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_5/stack_2?
)axis_angle_to_matrix/skew/strided_slice_5StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_5/stack:output:0:axis_angle_to_matrix/skew/strided_slice_5/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_5?
axis_angle_to_matrix/skew/Neg_2Neg2axis_angle_to_matrix/skew/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2!
axis_angle_to_matrix/skew/Neg_2?
/axis_angle_to_matrix/skew/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/axis_angle_to_matrix/skew/strided_slice_6/stack?
1axis_angle_to_matrix/skew/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1axis_angle_to_matrix/skew/strided_slice_6/stack_1?
1axis_angle_to_matrix/skew/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1axis_angle_to_matrix/skew/strided_slice_6/stack_2?
)axis_angle_to_matrix/skew/strided_slice_6StridedSlice*axis_angle_to_matrix/ExpandDims_1:output:08axis_angle_to_matrix/skew/strided_slice_6/stack:output:0:axis_angle_to_matrix/skew/strided_slice_6/stack_1:output:0:axis_angle_to_matrix/skew/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2+
)axis_angle_to_matrix/skew/strided_slice_6?
axis_angle_to_matrix/skew/stackPack!axis_angle_to_matrix/skew/Neg:y:02axis_angle_to_matrix/skew/strided_slice_2:output:02axis_angle_to_matrix/skew/strided_slice_3:output:0#axis_angle_to_matrix/skew/Neg_1:y:0#axis_angle_to_matrix/skew/Neg_2:y:02axis_angle_to_matrix/skew/strided_slice_6:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2!
axis_angle_to_matrix/skew/stack?
)axis_angle_to_matrix/skew/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)axis_angle_to_matrix/skew/Reshape_2/shape?
#axis_angle_to_matrix/skew/Reshape_2Reshape(axis_angle_to_matrix/skew/stack:output:02axis_angle_to_matrix/skew/Reshape_2/shape:output:0*
T0*#
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/Reshape_2?
!axis_angle_to_matrix/skew/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :	2#
!axis_angle_to_matrix/skew/mul_1/y?
axis_angle_to_matrix/skew/mul_1Mul0axis_angle_to_matrix/skew/strided_slice:output:0*axis_angle_to_matrix/skew/mul_1/y:output:0*
T0*
_output_shapes
: 2!
axis_angle_to_matrix/skew/mul_1?
)axis_angle_to_matrix/skew/ScatterNd/shapePack#axis_angle_to_matrix/skew/mul_1:z:0*
N*
T0*
_output_shapes
:2+
)axis_angle_to_matrix/skew/ScatterNd/shape?
#axis_angle_to_matrix/skew/ScatterNd	ScatterNd,axis_angle_to_matrix/skew/Reshape_1:output:0,axis_angle_to_matrix/skew/Reshape_2:output:02axis_angle_to_matrix/skew/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/ScatterNd?
+axis_angle_to_matrix/skew/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+axis_angle_to_matrix/skew/Reshape_3/shape/1?
+axis_angle_to_matrix/skew/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+axis_angle_to_matrix/skew/Reshape_3/shape/2?
)axis_angle_to_matrix/skew/Reshape_3/shapePack0axis_angle_to_matrix/skew/strided_slice:output:04axis_angle_to_matrix/skew/Reshape_3/shape/1:output:04axis_angle_to_matrix/skew/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)axis_angle_to_matrix/skew/Reshape_3/shape?
#axis_angle_to_matrix/skew/Reshape_3Reshape,axis_angle_to_matrix/skew/ScatterNd:output:02axis_angle_to_matrix/skew/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2%
#axis_angle_to_matrix/skew/Reshape_3?
axis_angle_to_matrix/mul_2Mulaxis_angle_to_matrix/Sin:y:0,axis_angle_to_matrix/skew/Reshape_3:output:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/mul_2?
axis_angle_to_matrix/add_2AddV2axis_angle_to_matrix/add_1:z:0axis_angle_to_matrix/mul_2:z:0*
T0*+
_output_shapes
:?????????2
axis_angle_to_matrix/add_2?
$axis_angle_to_matrix/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$axis_angle_to_matrix/concat/values_1?
 axis_angle_to_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 axis_angle_to_matrix/concat/axis?
axis_angle_to_matrix/concatConcatV2#axis_angle_to_matrix/Shape:output:0-axis_angle_to_matrix/concat/values_1:output:0)axis_angle_to_matrix/concat/axis:output:0*
N*
T0*
_output_shapes
:2
axis_angle_to_matrix/concat?
axis_angle_to_matrix/Reshape_1Reshapeaxis_angle_to_matrix/add_2:z:0$axis_angle_to_matrix/concat:output:0*
T0*/
_output_shapes
:?????????2 
axis_angle_to_matrix/Reshape_1?
tf_op_layer_Sub/Sub/yConst*
_output_shapes

:*
dtype0*=
value4B2"$  ??              ??              ??2
tf_op_layer_Sub/Sub/y?
tf_op_layer_Sub/SubSub'axis_angle_to_matrix/Reshape_1:output:0tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????2
tf_op_layer_Sub/Sub?
%tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%tf_op_layer_Reshape_1/Reshape_1/shape?
tf_op_layer_Reshape_1/Reshape_1Reshapetf_op_layer_Sub/Sub:z:0.tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*(
_output_shapes
:??????????2!
tf_op_layer_Reshape_1/Reshape_1?
(pose_encoder/dense/MatMul/ReadVariableOpReadVariableOp1pose_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(pose_encoder/dense/MatMul/ReadVariableOp?
pose_encoder/dense/MatMulMatMul(tf_op_layer_Reshape_1/Reshape_1:output:00pose_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
pose_encoder/dense/MatMul?
)pose_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2pose_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)pose_encoder/dense/BiasAdd/ReadVariableOp?
pose_encoder/dense/BiasAddBiasAdd#pose_encoder/dense/MatMul:product:01pose_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
pose_encoder/dense/BiasAdd?
pose_encoder/dense/ReluRelu#pose_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
pose_encoder/dense/Relu?
*pose_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3pose_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*pose_encoder/dense_1/MatMul/ReadVariableOp?
pose_encoder/dense_1/MatMulMatMul%pose_encoder/dense/Relu:activations:02pose_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
pose_encoder/dense_1/MatMul?
+pose_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4pose_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+pose_encoder/dense_1/BiasAdd/ReadVariableOp?
pose_encoder/dense_1/BiasAddBiasAdd%pose_encoder/dense_1/MatMul:product:03pose_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
pose_encoder/dense_1/BiasAddy
IdentityIdentity%pose_encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E:::::O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_dense_1_layer_call_fn_57103

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_562092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
l
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_56447

inputs
identitys
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape_1/shape?
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*
_cloned(*(
_output_shapes
:??????????2
	Reshape_1g
IdentityIdentityReshape_1:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_56433

inputs
identity?
Sub/yConst*
_output_shapes

:*
dtype0*=
value4B2"$  ??              ??              ??2
Sub/yr
SubSubinputsSub/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????2
Subc
IdentityIdentitySub:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_56847

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_565282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
O
3__inference_tf_op_layer_Reshape_layer_call_fn_56871

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_563052
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????E:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs
?
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56240
input_3
dense_56229
dense_56231
dense_1_56234
dense_1_56236
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_56229dense_56231*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_561832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56234dense_1_56236*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_562092!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_57021

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_model_1_layer_call_fn_56569	
poses
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallposesunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_565582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????E

_user_specified_nameposes:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_57038

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_56558

inputs
pose_encoder_56548
pose_encoder_56550
pose_encoder_56552
pose_encoder_56554
identity??$pose_encoder/StatefulPartitionedCall?
#tf_op_layer_Reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_563052%
#tf_op_layer_Reshape/PartitionedCall?
$axis_angle_to_matrix/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*X
fSRQ
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_564192&
$axis_angle_to_matrix/PartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall-axis_angle_to_matrix/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_564332!
tf_op_layer_Sub/PartitionedCall?
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_564472'
%tf_op_layer_Reshape_1/PartitionedCall?
$pose_encoder/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0pose_encoder_56548pose_encoder_56550pose_encoder_56552pose_encoder_56554*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562842&
$pose_encoder/StatefulPartitionedCall?
IdentityIdentity-pose_encoder/StatefulPartitionedCall:output:0%^pose_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::2L
$pose_encoder/StatefulPartitionedCall$pose_encoder/StatefulPartitionedCall:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?q
o
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_56977

axis_angle
identityH
ShapeShape
axis_angle*
T0*
_output_shapes
:2
Shapeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapes
ReshapeReshape
axis_angleReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
ReshapeR
Shape_1ShapeReshape:output:0*
T0*
_output_shapes
:2	
Shape_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/yg
addAddV2Reshape:output:0add/y:output:0*
T0*'
_output_shapes
:?????????2
add_
norm/mulMuladd:z:0add:z:0*
T0*'
_output_shapes
:?????????2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt{
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
2
norm/Squeezek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsnorm/Squeeze:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsv
truedivRealDivReshape:output:0ExpandDims:output:0*
T0*'
_output_shapes
:?????????2	
truedivo
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimstruediv:z:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDimsExpandDims:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_2^
CosCosExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
Cos^
SinSinExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
Sin?
outerBatchMatMulV2ExpandDims_1:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????*
adj_y(2
outera
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ??2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagf
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_3/dim?
ExpandDims_3
ExpandDimseye/diag:output:0ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_3f
Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :2
Tile/multiples/1f
Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :2
Tile/multiples/2?
Tile/multiplesPackstrided_slice:output:0Tile/multiples/1:output:0Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:2
Tile/multiplesz
TileTileExpandDims_3:output:0Tile/multiples:output:0*
T0*+
_output_shapes
:?????????2
Tile_
mulMulCos:y:0Tile:output:0*
T0*+
_output_shapes
:?????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Cos:y:0*
T0*+
_output_shapes
:?????????2
subd
mul_1Mulsub:z:0outer:output:0*
T0*+
_output_shapes
:?????????2
mul_1a
add_1AddV2mul:z:0	mul_1:z:0*
T0*+
_output_shapes
:?????????2
add_1]

skew/ShapeShapeExpandDims_1:output:0*
T0*
_output_shapes
:2

skew/Shape~
skew/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
skew/strided_slice/stack?
skew/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
skew/strided_slice/stack_1?
skew/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
skew/strided_slice/stack_2?
skew/strided_sliceStridedSliceskew/Shape:output:0!skew/strided_slice/stack:output:0#skew/strided_slice/stack_1:output:0#skew/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
skew/strided_slicey

skew/ConstConst*
_output_shapes
:*
dtype0*-
value$B""                  2

skew/Constf
skew/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
skew/range/startf
skew/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
skew/range/delta?

skew/rangeRangeskew/range/start:output:0skew/strided_slice:output:0skew/range/delta:output:0*#
_output_shapes
:?????????2

skew/rangeZ

skew/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2

skew/mul/ys
skew/mulMulskew/range:output:0skew/mul/y:output:0*
T0*#
_output_shapes
:?????????2

skew/muly
skew/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
skew/Reshape/shape?
skew/ReshapeReshapeskew/mul:z:0skew/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
skew/Reshape{
skew/addAddV2skew/Reshape:output:0skew/Const:output:0*
T0*'
_output_shapes
:?????????2

skew/add}
skew/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
skew/Reshape_1/shape?
skew/Reshape_1Reshapeskew/add:z:0skew/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
skew/Reshape_1?
skew/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_1/stack?
skew/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_1/stack_1?
skew/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_1/stack_2?
skew/strided_slice_1StridedSliceExpandDims_1:output:0#skew/strided_slice_1/stack:output:0%skew/strided_slice_1/stack_1:output:0%skew/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_1l
skew/NegNegskew/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

skew/Neg?
skew/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_2/stack?
skew/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_2/stack_1?
skew/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_2/stack_2?
skew/strided_slice_2StridedSliceExpandDims_1:output:0#skew/strided_slice_2/stack:output:0%skew/strided_slice_2/stack_1:output:0%skew/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_2?
skew/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_3/stack?
skew/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_3/stack_1?
skew/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_3/stack_2?
skew/strided_slice_3StridedSliceExpandDims_1:output:0#skew/strided_slice_3/stack:output:0%skew/strided_slice_3/stack_1:output:0%skew/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_3?
skew/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
skew/strided_slice_4/stack?
skew/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_4/stack_1?
skew/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_4/stack_2?
skew/strided_slice_4StridedSliceExpandDims_1:output:0#skew/strided_slice_4/stack:output:0%skew/strided_slice_4/stack_1:output:0%skew/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_4p

skew/Neg_1Negskew/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

skew/Neg_1?
skew/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_5/stack?
skew/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_5/stack_1?
skew/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_5/stack_2?
skew/strided_slice_5StridedSliceExpandDims_1:output:0#skew/strided_slice_5/stack:output:0%skew/strided_slice_5/stack_1:output:0%skew/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_5p

skew/Neg_2Negskew/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

skew/Neg_2?
skew/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
skew/strided_slice_6/stack?
skew/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
skew/strided_slice_6/stack_1?
skew/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
skew/strided_slice_6/stack_2?
skew/strided_slice_6StridedSliceExpandDims_1:output:0#skew/strided_slice_6/stack:output:0%skew/strided_slice_6/stack_1:output:0%skew/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
skew/strided_slice_6?

skew/stackPackskew/Neg:y:0skew/strided_slice_2:output:0skew/strided_slice_3:output:0skew/Neg_1:y:0skew/Neg_2:y:0skew/strided_slice_6:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2

skew/stack
skew/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
skew/Reshape_2/shape?
skew/Reshape_2Reshapeskew/stack:output:0skew/Reshape_2/shape:output:0*
T0*#
_output_shapes
:?????????2
skew/Reshape_2^
skew/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :	2
skew/mul_1/yt

skew/mul_1Mulskew/strided_slice:output:0skew/mul_1/y:output:0*
T0*
_output_shapes
: 2

skew/mul_1r
skew/ScatterNd/shapePackskew/mul_1:z:0*
N*
T0*
_output_shapes
:2
skew/ScatterNd/shape?
skew/ScatterNd	ScatterNdskew/Reshape_1:output:0skew/Reshape_2:output:0skew/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
skew/ScatterNdr
skew/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
skew/Reshape_3/shape/1r
skew/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
skew/Reshape_3/shape/2?
skew/Reshape_3/shapePackskew/strided_slice:output:0skew/Reshape_3/shape/1:output:0skew/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
skew/Reshape_3/shape?
skew/Reshape_3Reshapeskew/ScatterNd:output:0skew/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2
skew/Reshape_3m
mul_2MulSin:y:0skew/Reshape_3:output:0*
T0*+
_output_shapes
:?????????2
mul_2c
add_2AddV2	add_1:z:0	mul_2:z:0*
T0*+
_output_shapes
:?????????2
add_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Shape:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatw
	Reshape_1Reshape	add_2:z:0concat:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_1n
IdentityIdentityReshape_1:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:W S
+
_output_shapes
:?????????
$
_user_specified_name
axis_angle
??
?
 __inference__wrapped_model_56168	
poses=
9model_1_pose_encoder_dense_matmul_readvariableop_resource>
:model_1_pose_encoder_dense_biasadd_readvariableop_resource?
;model_1_pose_encoder_dense_1_matmul_readvariableop_resource@
<model_1_pose_encoder_dense_1_biasadd_readvariableop_resource
identity??
)model_1/tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2+
)model_1/tf_op_layer_Reshape/Reshape/shape?
#model_1/tf_op_layer_Reshape/ReshapeReshapeposes2model_1/tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*+
_output_shapes
:?????????2%
#model_1/tf_op_layer_Reshape/Reshape?
"model_1/axis_angle_to_matrix/ShapeShape,model_1/tf_op_layer_Reshape/Reshape:output:0*
T0*
_output_shapes
:2$
"model_1/axis_angle_to_matrix/Shape?
*model_1/axis_angle_to_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*model_1/axis_angle_to_matrix/Reshape/shape?
$model_1/axis_angle_to_matrix/ReshapeReshape,model_1/tf_op_layer_Reshape/Reshape:output:03model_1/axis_angle_to_matrix/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2&
$model_1/axis_angle_to_matrix/Reshape?
$model_1/axis_angle_to_matrix/Shape_1Shape-model_1/axis_angle_to_matrix/Reshape:output:0*
T0*
_output_shapes
:2&
$model_1/axis_angle_to_matrix/Shape_1?
0model_1/axis_angle_to_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/axis_angle_to_matrix/strided_slice/stack?
2model_1/axis_angle_to_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/axis_angle_to_matrix/strided_slice/stack_1?
2model_1/axis_angle_to_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/axis_angle_to_matrix/strided_slice/stack_2?
*model_1/axis_angle_to_matrix/strided_sliceStridedSlice-model_1/axis_angle_to_matrix/Shape_1:output:09model_1/axis_angle_to_matrix/strided_slice/stack:output:0;model_1/axis_angle_to_matrix/strided_slice/stack_1:output:0;model_1/axis_angle_to_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/axis_angle_to_matrix/strided_slice?
"model_1/axis_angle_to_matrix/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22$
"model_1/axis_angle_to_matrix/add/y?
 model_1/axis_angle_to_matrix/addAddV2-model_1/axis_angle_to_matrix/Reshape:output:0+model_1/axis_angle_to_matrix/add/y:output:0*
T0*'
_output_shapes
:?????????2"
 model_1/axis_angle_to_matrix/add?
%model_1/axis_angle_to_matrix/norm/mulMul$model_1/axis_angle_to_matrix/add:z:0$model_1/axis_angle_to_matrix/add:z:0*
T0*'
_output_shapes
:?????????2'
%model_1/axis_angle_to_matrix/norm/mul?
7model_1/axis_angle_to_matrix/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:29
7model_1/axis_angle_to_matrix/norm/Sum/reduction_indices?
%model_1/axis_angle_to_matrix/norm/SumSum)model_1/axis_angle_to_matrix/norm/mul:z:0@model_1/axis_angle_to_matrix/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2'
%model_1/axis_angle_to_matrix/norm/Sum?
&model_1/axis_angle_to_matrix/norm/SqrtSqrt.model_1/axis_angle_to_matrix/norm/Sum:output:0*
T0*'
_output_shapes
:?????????2(
&model_1/axis_angle_to_matrix/norm/Sqrt?
)model_1/axis_angle_to_matrix/norm/SqueezeSqueeze*model_1/axis_angle_to_matrix/norm/Sqrt:y:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
2+
)model_1/axis_angle_to_matrix/norm/Squeeze?
+model_1/axis_angle_to_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+model_1/axis_angle_to_matrix/ExpandDims/dim?
'model_1/axis_angle_to_matrix/ExpandDims
ExpandDims2model_1/axis_angle_to_matrix/norm/Squeeze:output:04model_1/axis_angle_to_matrix/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2)
'model_1/axis_angle_to_matrix/ExpandDims?
$model_1/axis_angle_to_matrix/truedivRealDiv-model_1/axis_angle_to_matrix/Reshape:output:00model_1/axis_angle_to_matrix/ExpandDims:output:0*
T0*'
_output_shapes
:?????????2&
$model_1/axis_angle_to_matrix/truediv?
-model_1/axis_angle_to_matrix/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model_1/axis_angle_to_matrix/ExpandDims_1/dim?
)model_1/axis_angle_to_matrix/ExpandDims_1
ExpandDims(model_1/axis_angle_to_matrix/truediv:z:06model_1/axis_angle_to_matrix/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2+
)model_1/axis_angle_to_matrix/ExpandDims_1?
-model_1/axis_angle_to_matrix/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model_1/axis_angle_to_matrix/ExpandDims_2/dim?
)model_1/axis_angle_to_matrix/ExpandDims_2
ExpandDims0model_1/axis_angle_to_matrix/ExpandDims:output:06model_1/axis_angle_to_matrix/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:?????????2+
)model_1/axis_angle_to_matrix/ExpandDims_2?
 model_1/axis_angle_to_matrix/CosCos2model_1/axis_angle_to_matrix/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2"
 model_1/axis_angle_to_matrix/Cos?
 model_1/axis_angle_to_matrix/SinSin2model_1/axis_angle_to_matrix/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2"
 model_1/axis_angle_to_matrix/Sin?
"model_1/axis_angle_to_matrix/outerBatchMatMulV22model_1/axis_angle_to_matrix/ExpandDims_1:output:02model_1/axis_angle_to_matrix/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????*
adj_y(2$
"model_1/axis_angle_to_matrix/outer?
%model_1/axis_angle_to_matrix/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ??2'
%model_1/axis_angle_to_matrix/eye/ones?
'model_1/axis_angle_to_matrix/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/axis_angle_to_matrix/eye/diag/k?
.model_1/axis_angle_to_matrix/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.model_1/axis_angle_to_matrix/eye/diag/num_rows?
.model_1/axis_angle_to_matrix/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.model_1/axis_angle_to_matrix/eye/diag/num_cols?
3model_1/axis_angle_to_matrix/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3model_1/axis_angle_to_matrix/eye/diag/padding_value?
%model_1/axis_angle_to_matrix/eye/diagMatrixDiagV3.model_1/axis_angle_to_matrix/eye/ones:output:00model_1/axis_angle_to_matrix/eye/diag/k:output:07model_1/axis_angle_to_matrix/eye/diag/num_rows:output:07model_1/axis_angle_to_matrix/eye/diag/num_cols:output:0<model_1/axis_angle_to_matrix/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2'
%model_1/axis_angle_to_matrix/eye/diag?
-model_1/axis_angle_to_matrix/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_1/axis_angle_to_matrix/ExpandDims_3/dim?
)model_1/axis_angle_to_matrix/ExpandDims_3
ExpandDims.model_1/axis_angle_to_matrix/eye/diag:output:06model_1/axis_angle_to_matrix/ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2+
)model_1/axis_angle_to_matrix/ExpandDims_3?
-model_1/axis_angle_to_matrix/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_1/axis_angle_to_matrix/Tile/multiples/1?
-model_1/axis_angle_to_matrix/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_1/axis_angle_to_matrix/Tile/multiples/2?
+model_1/axis_angle_to_matrix/Tile/multiplesPack3model_1/axis_angle_to_matrix/strided_slice:output:06model_1/axis_angle_to_matrix/Tile/multiples/1:output:06model_1/axis_angle_to_matrix/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:2-
+model_1/axis_angle_to_matrix/Tile/multiples?
!model_1/axis_angle_to_matrix/TileTile2model_1/axis_angle_to_matrix/ExpandDims_3:output:04model_1/axis_angle_to_matrix/Tile/multiples:output:0*
T0*+
_output_shapes
:?????????2#
!model_1/axis_angle_to_matrix/Tile?
 model_1/axis_angle_to_matrix/mulMul$model_1/axis_angle_to_matrix/Cos:y:0*model_1/axis_angle_to_matrix/Tile:output:0*
T0*+
_output_shapes
:?????????2"
 model_1/axis_angle_to_matrix/mul?
"model_1/axis_angle_to_matrix/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"model_1/axis_angle_to_matrix/sub/x?
 model_1/axis_angle_to_matrix/subSub+model_1/axis_angle_to_matrix/sub/x:output:0$model_1/axis_angle_to_matrix/Cos:y:0*
T0*+
_output_shapes
:?????????2"
 model_1/axis_angle_to_matrix/sub?
"model_1/axis_angle_to_matrix/mul_1Mul$model_1/axis_angle_to_matrix/sub:z:0+model_1/axis_angle_to_matrix/outer:output:0*
T0*+
_output_shapes
:?????????2$
"model_1/axis_angle_to_matrix/mul_1?
"model_1/axis_angle_to_matrix/add_1AddV2$model_1/axis_angle_to_matrix/mul:z:0&model_1/axis_angle_to_matrix/mul_1:z:0*
T0*+
_output_shapes
:?????????2$
"model_1/axis_angle_to_matrix/add_1?
'model_1/axis_angle_to_matrix/skew/ShapeShape2model_1/axis_angle_to_matrix/ExpandDims_1:output:0*
T0*
_output_shapes
:2)
'model_1/axis_angle_to_matrix/skew/Shape?
5model_1/axis_angle_to_matrix/skew/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5model_1/axis_angle_to_matrix/skew/strided_slice/stack?
7model_1/axis_angle_to_matrix/skew/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_1/axis_angle_to_matrix/skew/strided_slice/stack_1?
7model_1/axis_angle_to_matrix/skew/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_1/axis_angle_to_matrix/skew/strided_slice/stack_2?
/model_1/axis_angle_to_matrix/skew/strided_sliceStridedSlice0model_1/axis_angle_to_matrix/skew/Shape:output:0>model_1/axis_angle_to_matrix/skew/strided_slice/stack:output:0@model_1/axis_angle_to_matrix/skew/strided_slice/stack_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/model_1/axis_angle_to_matrix/skew/strided_slice?
'model_1/axis_angle_to_matrix/skew/ConstConst*
_output_shapes
:*
dtype0*-
value$B""                  2)
'model_1/axis_angle_to_matrix/skew/Const?
-model_1/axis_angle_to_matrix/skew/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_1/axis_angle_to_matrix/skew/range/start?
-model_1/axis_angle_to_matrix/skew/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-model_1/axis_angle_to_matrix/skew/range/delta?
'model_1/axis_angle_to_matrix/skew/rangeRange6model_1/axis_angle_to_matrix/skew/range/start:output:08model_1/axis_angle_to_matrix/skew/strided_slice:output:06model_1/axis_angle_to_matrix/skew/range/delta:output:0*#
_output_shapes
:?????????2)
'model_1/axis_angle_to_matrix/skew/range?
'model_1/axis_angle_to_matrix/skew/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2)
'model_1/axis_angle_to_matrix/skew/mul/y?
%model_1/axis_angle_to_matrix/skew/mulMul0model_1/axis_angle_to_matrix/skew/range:output:00model_1/axis_angle_to_matrix/skew/mul/y:output:0*
T0*#
_output_shapes
:?????????2'
%model_1/axis_angle_to_matrix/skew/mul?
/model_1/axis_angle_to_matrix/skew/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   21
/model_1/axis_angle_to_matrix/skew/Reshape/shape?
)model_1/axis_angle_to_matrix/skew/ReshapeReshape)model_1/axis_angle_to_matrix/skew/mul:z:08model_1/axis_angle_to_matrix/skew/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)model_1/axis_angle_to_matrix/skew/Reshape?
%model_1/axis_angle_to_matrix/skew/addAddV22model_1/axis_angle_to_matrix/skew/Reshape:output:00model_1/axis_angle_to_matrix/skew/Const:output:0*
T0*'
_output_shapes
:?????????2'
%model_1/axis_angle_to_matrix/skew/add?
1model_1/axis_angle_to_matrix/skew/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1model_1/axis_angle_to_matrix/skew/Reshape_1/shape?
+model_1/axis_angle_to_matrix/skew/Reshape_1Reshape)model_1/axis_angle_to_matrix/skew/add:z:0:model_1/axis_angle_to_matrix/skew/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2-
+model_1/axis_angle_to_matrix/skew/Reshape_1?
7model_1/axis_angle_to_matrix/skew/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7model_1/axis_angle_to_matrix/skew/strided_slice_1/stack?
9model_1/axis_angle_to_matrix/skew/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9model_1/axis_angle_to_matrix/skew/strided_slice_1/stack_1?
9model_1/axis_angle_to_matrix/skew/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/axis_angle_to_matrix/skew/strided_slice_1/stack_2?
1model_1/axis_angle_to_matrix/skew/strided_slice_1StridedSlice2model_1/axis_angle_to_matrix/ExpandDims_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice_1/stack:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_1/stack_1:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask23
1model_1/axis_angle_to_matrix/skew/strided_slice_1?
%model_1/axis_angle_to_matrix/skew/NegNeg:model_1/axis_angle_to_matrix/skew/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2'
%model_1/axis_angle_to_matrix/skew/Neg?
7model_1/axis_angle_to_matrix/skew/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7model_1/axis_angle_to_matrix/skew/strided_slice_2/stack?
9model_1/axis_angle_to_matrix/skew/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9model_1/axis_angle_to_matrix/skew/strided_slice_2/stack_1?
9model_1/axis_angle_to_matrix/skew/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/axis_angle_to_matrix/skew/strided_slice_2/stack_2?
1model_1/axis_angle_to_matrix/skew/strided_slice_2StridedSlice2model_1/axis_angle_to_matrix/ExpandDims_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice_2/stack:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_2/stack_1:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask23
1model_1/axis_angle_to_matrix/skew/strided_slice_2?
7model_1/axis_angle_to_matrix/skew/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7model_1/axis_angle_to_matrix/skew/strided_slice_3/stack?
9model_1/axis_angle_to_matrix/skew/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9model_1/axis_angle_to_matrix/skew/strided_slice_3/stack_1?
9model_1/axis_angle_to_matrix/skew/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/axis_angle_to_matrix/skew/strided_slice_3/stack_2?
1model_1/axis_angle_to_matrix/skew/strided_slice_3StridedSlice2model_1/axis_angle_to_matrix/ExpandDims_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice_3/stack:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_3/stack_1:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask23
1model_1/axis_angle_to_matrix/skew/strided_slice_3?
7model_1/axis_angle_to_matrix/skew/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7model_1/axis_angle_to_matrix/skew/strided_slice_4/stack?
9model_1/axis_angle_to_matrix/skew/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9model_1/axis_angle_to_matrix/skew/strided_slice_4/stack_1?
9model_1/axis_angle_to_matrix/skew/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/axis_angle_to_matrix/skew/strided_slice_4/stack_2?
1model_1/axis_angle_to_matrix/skew/strided_slice_4StridedSlice2model_1/axis_angle_to_matrix/ExpandDims_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice_4/stack:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_4/stack_1:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask23
1model_1/axis_angle_to_matrix/skew/strided_slice_4?
'model_1/axis_angle_to_matrix/skew/Neg_1Neg:model_1/axis_angle_to_matrix/skew/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2)
'model_1/axis_angle_to_matrix/skew/Neg_1?
7model_1/axis_angle_to_matrix/skew/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7model_1/axis_angle_to_matrix/skew/strided_slice_5/stack?
9model_1/axis_angle_to_matrix/skew/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9model_1/axis_angle_to_matrix/skew/strided_slice_5/stack_1?
9model_1/axis_angle_to_matrix/skew/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/axis_angle_to_matrix/skew/strided_slice_5/stack_2?
1model_1/axis_angle_to_matrix/skew/strided_slice_5StridedSlice2model_1/axis_angle_to_matrix/ExpandDims_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice_5/stack:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_5/stack_1:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask23
1model_1/axis_angle_to_matrix/skew/strided_slice_5?
'model_1/axis_angle_to_matrix/skew/Neg_2Neg:model_1/axis_angle_to_matrix/skew/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2)
'model_1/axis_angle_to_matrix/skew/Neg_2?
7model_1/axis_angle_to_matrix/skew/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7model_1/axis_angle_to_matrix/skew/strided_slice_6/stack?
9model_1/axis_angle_to_matrix/skew/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9model_1/axis_angle_to_matrix/skew/strided_slice_6/stack_1?
9model_1/axis_angle_to_matrix/skew/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/axis_angle_to_matrix/skew/strided_slice_6/stack_2?
1model_1/axis_angle_to_matrix/skew/strided_slice_6StridedSlice2model_1/axis_angle_to_matrix/ExpandDims_1:output:0@model_1/axis_angle_to_matrix/skew/strided_slice_6/stack:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_6/stack_1:output:0Bmodel_1/axis_angle_to_matrix/skew/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask23
1model_1/axis_angle_to_matrix/skew/strided_slice_6?
'model_1/axis_angle_to_matrix/skew/stackPack)model_1/axis_angle_to_matrix/skew/Neg:y:0:model_1/axis_angle_to_matrix/skew/strided_slice_2:output:0:model_1/axis_angle_to_matrix/skew/strided_slice_3:output:0+model_1/axis_angle_to_matrix/skew/Neg_1:y:0+model_1/axis_angle_to_matrix/skew/Neg_2:y:0:model_1/axis_angle_to_matrix/skew/strided_slice_6:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2)
'model_1/axis_angle_to_matrix/skew/stack?
1model_1/axis_angle_to_matrix/skew/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1model_1/axis_angle_to_matrix/skew/Reshape_2/shape?
+model_1/axis_angle_to_matrix/skew/Reshape_2Reshape0model_1/axis_angle_to_matrix/skew/stack:output:0:model_1/axis_angle_to_matrix/skew/Reshape_2/shape:output:0*
T0*#
_output_shapes
:?????????2-
+model_1/axis_angle_to_matrix/skew/Reshape_2?
)model_1/axis_angle_to_matrix/skew/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :	2+
)model_1/axis_angle_to_matrix/skew/mul_1/y?
'model_1/axis_angle_to_matrix/skew/mul_1Mul8model_1/axis_angle_to_matrix/skew/strided_slice:output:02model_1/axis_angle_to_matrix/skew/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'model_1/axis_angle_to_matrix/skew/mul_1?
1model_1/axis_angle_to_matrix/skew/ScatterNd/shapePack+model_1/axis_angle_to_matrix/skew/mul_1:z:0*
N*
T0*
_output_shapes
:23
1model_1/axis_angle_to_matrix/skew/ScatterNd/shape?
+model_1/axis_angle_to_matrix/skew/ScatterNd	ScatterNd4model_1/axis_angle_to_matrix/skew/Reshape_1:output:04model_1/axis_angle_to_matrix/skew/Reshape_2:output:0:model_1/axis_angle_to_matrix/skew/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2-
+model_1/axis_angle_to_matrix/skew/ScatterNd?
3model_1/axis_angle_to_matrix/skew/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3model_1/axis_angle_to_matrix/skew/Reshape_3/shape/1?
3model_1/axis_angle_to_matrix/skew/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3model_1/axis_angle_to_matrix/skew/Reshape_3/shape/2?
1model_1/axis_angle_to_matrix/skew/Reshape_3/shapePack8model_1/axis_angle_to_matrix/skew/strided_slice:output:0<model_1/axis_angle_to_matrix/skew/Reshape_3/shape/1:output:0<model_1/axis_angle_to_matrix/skew/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:23
1model_1/axis_angle_to_matrix/skew/Reshape_3/shape?
+model_1/axis_angle_to_matrix/skew/Reshape_3Reshape4model_1/axis_angle_to_matrix/skew/ScatterNd:output:0:model_1/axis_angle_to_matrix/skew/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2-
+model_1/axis_angle_to_matrix/skew/Reshape_3?
"model_1/axis_angle_to_matrix/mul_2Mul$model_1/axis_angle_to_matrix/Sin:y:04model_1/axis_angle_to_matrix/skew/Reshape_3:output:0*
T0*+
_output_shapes
:?????????2$
"model_1/axis_angle_to_matrix/mul_2?
"model_1/axis_angle_to_matrix/add_2AddV2&model_1/axis_angle_to_matrix/add_1:z:0&model_1/axis_angle_to_matrix/mul_2:z:0*
T0*+
_output_shapes
:?????????2$
"model_1/axis_angle_to_matrix/add_2?
,model_1/axis_angle_to_matrix/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_1/axis_angle_to_matrix/concat/values_1?
(model_1/axis_angle_to_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/axis_angle_to_matrix/concat/axis?
#model_1/axis_angle_to_matrix/concatConcatV2+model_1/axis_angle_to_matrix/Shape:output:05model_1/axis_angle_to_matrix/concat/values_1:output:01model_1/axis_angle_to_matrix/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#model_1/axis_angle_to_matrix/concat?
&model_1/axis_angle_to_matrix/Reshape_1Reshape&model_1/axis_angle_to_matrix/add_2:z:0,model_1/axis_angle_to_matrix/concat:output:0*
T0*/
_output_shapes
:?????????2(
&model_1/axis_angle_to_matrix/Reshape_1?
model_1/tf_op_layer_Sub/Sub/yConst*
_output_shapes

:*
dtype0*=
value4B2"$  ??              ??              ??2
model_1/tf_op_layer_Sub/Sub/y?
model_1/tf_op_layer_Sub/SubSub/model_1/axis_angle_to_matrix/Reshape_1:output:0&model_1/tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????2
model_1/tf_op_layer_Sub/Sub?
-model_1/tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2/
-model_1/tf_op_layer_Reshape_1/Reshape_1/shape?
'model_1/tf_op_layer_Reshape_1/Reshape_1Reshapemodel_1/tf_op_layer_Sub/Sub:z:06model_1/tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*(
_output_shapes
:??????????2)
'model_1/tf_op_layer_Reshape_1/Reshape_1?
0model_1/pose_encoder/dense/MatMul/ReadVariableOpReadVariableOp9model_1_pose_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0model_1/pose_encoder/dense/MatMul/ReadVariableOp?
!model_1/pose_encoder/dense/MatMulMatMul0model_1/tf_op_layer_Reshape_1/Reshape_1:output:08model_1/pose_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_1/pose_encoder/dense/MatMul?
1model_1/pose_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp:model_1_pose_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_1/pose_encoder/dense/BiasAdd/ReadVariableOp?
"model_1/pose_encoder/dense/BiasAddBiasAdd+model_1/pose_encoder/dense/MatMul:product:09model_1/pose_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_1/pose_encoder/dense/BiasAdd?
model_1/pose_encoder/dense/ReluRelu+model_1/pose_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
model_1/pose_encoder/dense/Relu?
2model_1/pose_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_pose_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype024
2model_1/pose_encoder/dense_1/MatMul/ReadVariableOp?
#model_1/pose_encoder/dense_1/MatMulMatMul-model_1/pose_encoder/dense/Relu:activations:0:model_1/pose_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2%
#model_1/pose_encoder/dense_1/MatMul?
3model_1/pose_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_pose_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3model_1/pose_encoder/dense_1/BiasAdd/ReadVariableOp?
$model_1/pose_encoder/dense_1/BiasAddBiasAdd-model_1/pose_encoder/dense_1/MatMul:product:0;model_1/pose_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2&
$model_1/pose_encoder/dense_1/BiasAdd?
IdentityIdentity-model_1/pose_encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E:::::N J
'
_output_shapes
:?????????E

_user_specified_nameposes:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_56491	
poses
pose_encoder_56481
pose_encoder_56483
pose_encoder_56485
pose_encoder_56487
identity??$pose_encoder/StatefulPartitionedCall?
#tf_op_layer_Reshape/PartitionedCallPartitionedCallposes*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_563052%
#tf_op_layer_Reshape/PartitionedCall?
$axis_angle_to_matrix/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*X
fSRQ
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_564192&
$axis_angle_to_matrix/PartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall-axis_angle_to_matrix/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_564332!
tf_op_layer_Sub/PartitionedCall?
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_564472'
%tf_op_layer_Reshape_1/PartitionedCall?
$pose_encoder/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0pose_encoder_56481pose_encoder_56483pose_encoder_56485pose_encoder_56487*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562572&
$pose_encoder/StatefulPartitionedCall?
IdentityIdentity-pose_encoder/StatefulPartitionedCall:output:0%^pose_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::2L
$pose_encoder/StatefulPartitionedCall$pose_encoder/StatefulPartitionedCall:N J
'
_output_shapes
:?????????E

_user_specified_nameposes:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_56183

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_model_1_layer_call_fn_56539	
poses
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallposesunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_565282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????E

_user_specified_nameposes:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56284

inputs
dense_56273
dense_56275
dense_1_56278
dense_1_56280
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56273dense_56275*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_561832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56278dense_1_56280*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_562092!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__traced_save_57142
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_12bf72ab4bf6464ba10038d378ffee73/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*;
_input_shapes*
(: :
??:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?
?
,__inference_pose_encoder_layer_call_fn_57064

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_pose_encoder_layer_call_fn_56268
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_56209

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_56528

inputs
pose_encoder_56518
pose_encoder_56520
pose_encoder_56522
pose_encoder_56524
identity??$pose_encoder/StatefulPartitionedCall?
#tf_op_layer_Reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_563052%
#tf_op_layer_Reshape/PartitionedCall?
$axis_angle_to_matrix/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*X
fSRQ
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_564192&
$axis_angle_to_matrix/PartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall-axis_angle_to_matrix/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_564332!
tf_op_layer_Sub/PartitionedCall?
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_564472'
%tf_op_layer_Reshape_1/PartitionedCall?
$pose_encoder/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0pose_encoder_56518pose_encoder_56520pose_encoder_56522pose_encoder_56524*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562572&
$pose_encoder/StatefulPartitionedCall?
IdentityIdentity-pose_encoder/StatefulPartitionedCall:output:0%^pose_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::2L
$pose_encoder/StatefulPartitionedCall$pose_encoder/StatefulPartitionedCall:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56257

inputs
dense_56246
dense_56248
dense_1_56251
dense_1_56253
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56246dense_56248*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_561832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56251dense_1_56253*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_562092!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_56866

inputs
identitys
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????E:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_57094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_signature_wrapper_56584	
poses
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallposesunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*)
f$R"
 __inference__wrapped_model_561682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????E

_user_specified_nameposes:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_pose_encoder_layer_call_fn_56295
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_pose_encoder_layer_call_and_return_conditional_losses_562842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
!__inference__traced_restore_57166
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_model_1_layer_call_fn_56860

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_565582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????E::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
l
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_56999

inputs
identitys
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape_1/shape?
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*
_cloned(*(
_output_shapes
:??????????2
	Reshape_1g
IdentityIdentityReshape_1:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
poses.
serving_default_poses:0?????????E@
pose_encoder0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?5
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
*W&call_and_return_all_conditional_losses
X__call__
Y_default_save_signature"?3
_tf_keras_model?3{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 69]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "poses"}, "name": "poses", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["poses", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 23, 3]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["poses", 0, 0, {}]]]}, {"class_name": "AxisAngleToMatrix", "config": {"name": "axis_angle_to_matrix", "trainable": true, "dtype": "float32"}, "name": "axis_angle_to_matrix", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["axis_angle_to_matrix/Identity", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["axis_angle_to_matrix", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["Sub", "Reshape_1/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 207]}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "pose_encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 207]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "name": "pose_encoder", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}]]]}], "input_layers": [["poses", 0, 0]], "output_layers": [["pose_encoder", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 69]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 69]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "poses"}, "name": "poses", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["poses", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 23, 3]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["poses", 0, 0, {}]]]}, {"class_name": "AxisAngleToMatrix", "config": {"name": "axis_angle_to_matrix", "trainable": true, "dtype": "float32"}, "name": "axis_angle_to_matrix", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["axis_angle_to_matrix/Identity", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["axis_angle_to_matrix", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["Sub", "Reshape_1/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 207]}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "pose_encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 207]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "name": "pose_encoder", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}]]]}], "input_layers": [["poses", 0, 0]], "output_layers": [["pose_encoder", 1, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "poses", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 69]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 69]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "poses"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["poses", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 23, 3]}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"class_name": "AxisAngleToMatrix", "name": "axis_angle_to_matrix", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "axis_angle_to_matrix", "trainable": true, "dtype": "float32"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["axis_angle_to_matrix/Identity", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["Sub", "Reshape_1/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 207]}}}
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
 regularization_losses
!trainable_variables
"	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_model?{"class_name": "Model", "name": "pose_encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "pose_encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 207]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 207]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "pose_encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 207]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}}
<
#0
$1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
#0
$1
%2
&3"
trackable_list_wrapper
?
'layer_metrics
(layer_regularization_losses
)non_trainable_variables
*metrics
	variables
regularization_losses

+layers
	trainable_variables
X__call__
Y_default_save_signature
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
,
dserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,layer_metrics
-layer_regularization_losses
.non_trainable_variables
/metrics
	variables
regularization_losses

0layers
trainable_variables
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1layer_metrics
2layer_regularization_losses
3non_trainable_variables
4metrics
	variables
regularization_losses

5layers
trainable_variables
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6layer_metrics
7layer_regularization_losses
8non_trainable_variables
9metrics
	variables
regularization_losses

:layers
trainable_variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
>metrics
	variables
regularization_losses

?layers
trainable_variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 207]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 207]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?

#kernel
$bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
*e&call_and_return_all_conditional_losses
f__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 207}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 207]}}
?

%kernel
&bias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
*g&call_and_return_all_conditional_losses
h__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
<
#0
$1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
#0
$1
%2
&3"
trackable_list_wrapper
?
Hlayer_metrics
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
	variables
 regularization_losses

Llayers
!trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 :
??2dense/kernel
:?2
dense/bias
!:	?
2dense_1/kernel
:
2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
Mlayer_metrics
Nlayer_regularization_losses
Onon_trainable_variables
Pmetrics
@	variables
Aregularization_losses

Qlayers
Btrainable_variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
Rlayer_metrics
Slayer_regularization_losses
Tnon_trainable_variables
Umetrics
D	variables
Eregularization_losses

Vlayers
Ftrainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_56709
B__inference_model_1_layer_call_and_return_conditional_losses_56508
B__inference_model_1_layer_call_and_return_conditional_losses_56834
B__inference_model_1_layer_call_and_return_conditional_losses_56491?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_1_layer_call_fn_56569
'__inference_model_1_layer_call_fn_56860
'__inference_model_1_layer_call_fn_56539
'__inference_model_1_layer_call_fn_56847?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_56168?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
poses?????????E
?2?
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_56866?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_tf_op_layer_Reshape_layer_call_fn_56871?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_56977?
???
FullArgSpec!
args?
jself
j
axis_angle
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_axis_angle_to_matrix_layer_call_fn_56982?
???
FullArgSpec!
args?
jself
j
axis_angle
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_56988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_tf_op_layer_Sub_layer_call_fn_56993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_56999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_tf_op_layer_Reshape_1_layer_call_fn_57004?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_57038
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56240
G__inference_pose_encoder_layer_call_and_return_conditional_losses_57021
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56226?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_pose_encoder_layer_call_fn_56268
,__inference_pose_encoder_layer_call_fn_56295
,__inference_pose_encoder_layer_call_fn_57051
,__inference_pose_encoder_layer_call_fn_57064?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
0B.
#__inference_signature_wrapper_56584poses
?2?
@__inference_dense_layer_call_and_return_conditional_losses_57075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_57084?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_57094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_57103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_56168s#$%&.?+
$?!
?
poses?????????E
? ";?8
6
pose_encoder&?#
pose_encoder?????????
?
O__inference_axis_angle_to_matrix_layer_call_and_return_conditional_losses_56977h7?4
-?*
(?%

axis_angle?????????
? "-?*
#? 
0?????????
? ?
4__inference_axis_angle_to_matrix_layer_call_fn_56982[7?4
-?*
(?%

axis_angle?????????
? " ???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_57094]%&0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? {
'__inference_dense_1_layer_call_fn_57103P%&0?-
&?#
!?
inputs??????????
? "??????????
?
@__inference_dense_layer_call_and_return_conditional_losses_57075^#$0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
%__inference_dense_layer_call_fn_57084Q#$0?-
&?#
!?
inputs??????????
? "????????????
B__inference_model_1_layer_call_and_return_conditional_losses_56491e#$%&6?3
,?)
?
poses?????????E
p

 
? "%?"
?
0?????????

? ?
B__inference_model_1_layer_call_and_return_conditional_losses_56508e#$%&6?3
,?)
?
poses?????????E
p 

 
? "%?"
?
0?????????

? ?
B__inference_model_1_layer_call_and_return_conditional_losses_56709f#$%&7?4
-?*
 ?
inputs?????????E
p

 
? "%?"
?
0?????????

? ?
B__inference_model_1_layer_call_and_return_conditional_losses_56834f#$%&7?4
-?*
 ?
inputs?????????E
p 

 
? "%?"
?
0?????????

? ?
'__inference_model_1_layer_call_fn_56539X#$%&6?3
,?)
?
poses?????????E
p

 
? "??????????
?
'__inference_model_1_layer_call_fn_56569X#$%&6?3
,?)
?
poses?????????E
p 

 
? "??????????
?
'__inference_model_1_layer_call_fn_56847Y#$%&7?4
-?*
 ?
inputs?????????E
p

 
? "??????????
?
'__inference_model_1_layer_call_fn_56860Y#$%&7?4
-?*
 ?
inputs?????????E
p 

 
? "??????????
?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56226h#$%&9?6
/?,
"?
input_3??????????
p

 
? "%?"
?
0?????????

? ?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_56240h#$%&9?6
/?,
"?
input_3??????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_57021g#$%&8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????

? ?
G__inference_pose_encoder_layer_call_and_return_conditional_losses_57038g#$%&8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????

? ?
,__inference_pose_encoder_layer_call_fn_56268[#$%&9?6
/?,
"?
input_3??????????
p

 
? "??????????
?
,__inference_pose_encoder_layer_call_fn_56295[#$%&9?6
/?,
"?
input_3??????????
p 

 
? "??????????
?
,__inference_pose_encoder_layer_call_fn_57051Z#$%&8?5
.?+
!?
inputs??????????
p

 
? "??????????
?
,__inference_pose_encoder_layer_call_fn_57064Z#$%&8?5
.?+
!?
inputs??????????
p 

 
? "??????????
?
#__inference_signature_wrapper_56584|#$%&7?4
? 
-?*
(
poses?
poses?????????E";?8
6
pose_encoder&?#
pose_encoder?????????
?
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_56999a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
5__inference_tf_op_layer_Reshape_1_layer_call_fn_57004T7?4
-?*
(?%
inputs?????????
? "????????????
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_56866\/?,
%?"
 ?
inputs?????????E
? ")?&
?
0?????????
? ?
3__inference_tf_op_layer_Reshape_layer_call_fn_56871O/?,
%?"
 ?
inputs?????????E
? "???????????
J__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_56988h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
/__inference_tf_op_layer_Sub_layer_call_fn_56993[7?4
-?*
(?%
inputs?????????
? " ??????????