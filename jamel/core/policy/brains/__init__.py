from .naive.brain import NaiveBrain
from .explicit_memory.brain import BrainWithExplicitMemory
from .parametric_memory.brain import BrainWithParametricMemory
# 感觉 brain 和 memory 的耦合度很高，应该把这两个东西合并在一起？
# 实则不然！Memory 作为一个重要的模块是不能放在 Policy 下面的。只是我们现在这个版本的实现里，Memory 的模型和 Policy 的模型是同一个，在其他的场景中，我们完全有可能使用一个 Memory 模型和一个 Policy 模型。