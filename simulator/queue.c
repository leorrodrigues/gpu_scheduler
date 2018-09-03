char *basic_get(amqp_connection_state_t connection_state_,
                const char *queue_name_, uint64_t *out_body_size_) {
  amqp_rpc_reply_t rpc_reply;
  amqp_time_t deadline;
  struct timeval timeout = {5, 0};
  int time_rc = amqp_time_from_now(&deadline, &timeout);
  assert(time_rc == AMQP_STATUS_OK);

  do {
    rpc_reply = amqp_basic_get(connection_state_, fixed_channel_id,
                               amqp_cstring_bytes(queue_name_), /*no_ack*/ 1);
  } while (rpc_reply.reply_type == AMQP_RESPONSE_NORMAL &&
           rpc_reply.reply.id == AMQP_BASIC_GET_EMPTY_METHOD &&
           amqp_time_has_past(deadline) == AMQP_STATUS_OK);

  assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);
  assert(rpc_reply.reply.id == AMQP_BASIC_GET_OK_METHOD);

  amqp_message_t message;
  rpc_reply =
      amqp_read_message(connection_state_, fixed_channel_id, &message, 0);
  assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

  char *body = malloc(message.body.len);
  memcpy(body, message.body.bytes, message.body.len);
  *out_body_size_ = message.body.len;
  amqp_destroy_message(&message);

  return body;
}

void publish_and_basic_get_message(const char *msg_to_publish) {
  amqp_connection_state_t connection_state = setup_connection_and_channel();

  queue_declare(connection_state, test_queue_name);
  basic_publish(connection_state, msg_to_publish);

  uint64_t body_size;
  char *msg = basic_get(connection_state, test_queue_name, &body_size);

  assert(body_size == strlen(msg_to_publish));
  assert(strncmp(msg_to_publish, msg, body_size) == 0);
  free(msg);

  close_and_destroy_connection(connection_state);
}

char *consume_message(amqp_connection_state_t connection_state_,
                      const char *queue_name_, uint64_t *out_body_size_) {
  amqp_basic_consume_ok_t *result =
      amqp_basic_consume(connection_state_, fixed_channel_id,
                         amqp_cstring_bytes(queue_name_), amqp_empty_bytes,
                         /*no_local*/ 0,
                         /*no_ack*/ 1,
                         /*exclusive*/ 0, amqp_empty_table);
  assert(result != NULL);

  amqp_envelope_t envelope;
  struct timeval timeout = {5, 0};
  amqp_rpc_reply_t rpc_reply =
      amqp_consume_message(connection_state_, &envelope, &timeout, 0);
  assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

  *out_body_size_ = envelope.message.body.len;
  char *body = malloc(*out_body_size_);
  if (*out_body_size_) {
    memcpy(body, envelope.message.body.bytes, *out_body_size_);
  }

  amqp_destroy_envelope(&envelope);
  return body;
}

void publish_and_consume_message(const char *msg_to_publish) {
  amqp_connection_state_t connection_state = setup_connection_and_channel();

  queue_declare(connection_state, test_queue_name);
  basic_publish(connection_state, msg_to_publish);

  uint64_t body_size;
  char *msg = consume_message(connection_state, test_queue_name, &body_size);

  assert(body_size == strlen(msg_to_publish));
  assert(strncmp(msg_to_publish, msg, body_size) == 0);
  free(msg);

  close_and_destroy_connection(connection_state);
}

int main(void) {
  publish_and_basic_get_message("");
  publish_and_basic_get_message("TEST");

  publish_and_consume_message("");
  publish_and_consume_message("TEST");

  return 0;
}
