/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/radix_tree.h
 */
#ifndef MLC_LLM_SERVE_RADIX_TREE_H_
#define MLC_LLM_SERVE_RADIX_TREE_H_
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/object.h>

#include <unordered_map>
#include <unordered_set>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The paged radix tree data structure.
 */
class PagedRadixTreeObj : public Object {
 public:
  /*!
   * \brief Get a sequence's all tokens.
   * \param seq_id The sequence ID for index.
   * \return The sequence tokens.
   * \throw Error if sequence ID is not valid.
   */
  virtual IntTuple GetSequence(int64_t seq_id) = 0;

  /*!
   * \brief Get all sequences with longest common prefix with give prefix tokens.
   * \param tokens The prefix tokens for reference.
   * \return The pair of matched prefix length and the array of matched sequences indices.
   */
  virtual std::pair<size_t, std::vector<int64_t>> MatchPrefix(IntTuple tokens) = 0;

  /*!
   * \brief Get a sequence's length.
   * \param seq_id The sequence ID for index.
   * \return The sequence length.
   * \throw Error if sequence ID is not valid.
   */
  virtual size_t GetSequenceLength(int64_t seq_id) = 0;

  /*!
   * \brief Fork a sequence from parent sequence at given position.
   * \param seq_id The new sequence ID.
   * \param parent_seq_id The parent sequence ID to fork from.
   * \param forked_offset The position of parent sequence to fork at.
   * The valid value is [1, length of forked sequence]. If the position equals the length of forked
   * sequence, the new sequence will copy the entire forked sequence.
   * \throw Error if sequence ID or
   * forked postion is not valid.
   */
  virtual void ForkSequence(int64_t seq_id, int64_t parent_seq_id, size_t forked_offset) = 0;

  /*!
   * \brief Add an empty sequence at root.
   * \param seq_id The new sequence ID.
   * \throw Error if sequence ID is not valid.
   */
  virtual void AddSequence(int64_t seq_id) = 0;

  /*!
   * \brief Extend a sequence with given tokens.
   * \param seq_id The sequence ID for index.
   * \param tokens The given tokens to extend.
   * \throw Error if sequence ID is not valid.
   */
  virtual void ExtendSequence(int64_t seq_id, IntTuple tokens) = 0;

  /*!
   * \brief Remove a sequence.
   * \param seq_id The sequence ID to remove.
   * \throw Error if sequence ID is not valid.
   */
  virtual void RemoveSequence(int64_t seq_id) = 0;

  /*!
   * \brief Get the remaining token capacity of the paged radix tree.
   * \return The the remaining token capacity of the paged radix tree.
   */
  virtual size_t FreeCapacity() = 0;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mlc.serve.PagedRadixTree";
  TVM_DECLARE_BASE_OBJECT_INFO(PagedRadixTreeObj, Object)
};

TVM_REGISTER_OBJECT_TYPE(PagedRadixTreeObj);

class PagedRadixTree : public ObjectRef {
 public:
  /*!
   * \brief Constructor of paged radix tree.
   * \param num_pages The number of radix tree pages.
   * \param page_size The page size of each radix tree page.
   * \param num_seqs The maximum number of sequence ID.
   */
  PagedRadixTree(size_t num_pages, size_t page_size, size_t num_seqs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PagedRadixTree, ObjectRef, PagedRadixTreeObj);
};
}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_RADIX_TREE_H_
